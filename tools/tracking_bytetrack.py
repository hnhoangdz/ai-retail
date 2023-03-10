import argparse
import os
import os.path as osp
import time
import cv2
import torch
import logging
import numpy as np

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from config.config_detection import Yolov8Config, PoseConfig, Yolov7Config
from config.config_common import objects, AREA, INDEX_FRAME_IGNORE, NUM_FRAME_SET, Visualization
from tracking.mmpose.mmpose.apis import (inference_top_down_pose_model, init_pose_model)
from mmpose.datasets import DatasetInfo
from ultils.logger import set_logger
from ultils.behavior import PoseBehavior
from ultils.common import convert2Real, visual_area, get_monitor_size
from ultils.object import Item, Human, Box, Polygon, KeyPoint, Point, Frame
from ultils.preprocess import Yolov8Detector, Yolov7Detector, load_model

level = logging.WARNING
logger = set_logger(level=level)

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="show video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser
# write function show video cv2 python

    # Open the video file or camera
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        # If the frame was read successfully
        if ret:
            # Display the frame
            cv2.imshow('Video', frame)

            # Wait for 25ms and check if the user pressed the 'q' key
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def imageflow_demo(predictor, items_predictor, pose_model, dataset, dataset_info, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    result_in_num_frame_set = []
    results = []
    behavior = PoseBehavior()
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        frame = Frame(id=frame_id, img=frame, ratio=1)
        if ret_val:
            outputs, img_info = predictor.inference(frame.img, timer)
            items_dets, fps_items = items_predictor.detect(frame.img)


            # PROCESS PERSON
            if outputs[0] is not None:
                # import ipdb; ipdb.set_trace()
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                human_boxes = []
                for target in online_targets:
                    human_boxes.append({
                        'bbox':np.append(target.tlbr, target.score),
                        'tlwh':target.tlwh,
                        'track_id':target.track_id
                    })

                # inference pose in body human
                pose_results = inference_top_down_pose_model(
                    pose_model,
                    frame.img,
                    human_boxes,
                    bbox_thr=PoseConfig.bbox_thres,
                    format='xyxy',
                    dataset=dataset,
                    dataset_info=dataset_info,
                    return_heatmap=PoseConfig.return_heatmap,
                    outputs=PoseConfig.ouputs
                )
                human_boxes.clear()
                for each in pose_results[0]:
                    human = Human(
                        track_id=each['track_id'],
                        id_object=None,
                        name_object="Person",
                        box=Box(tl=Point(x=each['bbox'][0], y=each['bbox'][1]), br=Point(x=each['bbox'][2], y=each['bbox'][3])),
                        conf=each['bbox'][4]
                    )
                    # append keypoint hand and leg 
                    for i, kp in enumerate(each['keypoints'].tolist()):
                        config_kp_part = [
                            PoseConfig.id_pose_lefthand, PoseConfig.id_pose_righthand, PoseConfig.id_pose_rightleg, PoseConfig.id_pose_leftleg
                        ]

                        body_part = [human.left_hand_kp, human.right_hand_kp, human.right_leg_kp, human.left_leg_kp]

                        # collect left_hand_keypoints
                        for config_kp, kp_body in zip(config_kp_part, body_part): 
                            for id_pose in  config_kp:
                                # get id pose in each_human fit pose left hand 
                                if i == id_pose:
                                    meta_one_point = PoseConfig.keypoint_definition[str(id_pose)]
                                    each_pose = KeyPoint(x=kp[0], y=kp[1], 
                                                        conf=kp[2], id_=i, 
                                                        name=meta_one_point['name'],
                                                        color=meta_one_point['color'],
                                                        type_=meta_one_point['type'],
                                                        swap=meta_one_point['swap'])
                                    kp_body.append(each_pose)
                    frame.humans.append(human)  
                # add result of nums frames continuous

                frame.items = [each for each in items_dets]
                result_in_num_frame_set.append(frame)
                if len(result_in_num_frame_set) > NUM_FRAME_SET:
                    result_in_num_frame_set = result_in_num_frame_set[-1*NUM_FRAME_SET:]
                # VISUALIZE AREA
                if Visualization.allow_visual_area:
                    visual_area(image=frame)

                if frame_id > INDEX_FRAME_IGNORE:
                    behavior.get_item(rs_num_frame_consecutive=result_in_num_frame_set)
                    behavior.to_pay(rs_num_frame_consecutive=result_in_num_frame_set, all_human_in_retail=behavior.result)

                    # VISUALIZE HUMAN, ITEM
                    for human in behavior.result:
                        if Visualization.visual_box_human:
                            human.visual(image=frame.img, label=f"id: {human.track_id}")
                        if Visualization.visual_box_item:
                            for item in human.item_holding:
                                item.visual(image=frame.img, label=item.name_object)
              
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n")
                timer.toc()
                online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time)
            else:
                timer.toc()
                online_im = img_info['raw_img']
            # cv2.imshow("", online_im)
            # cv2.waitKey(0)
            
            mo_wid, mo_hei = get_monitor_size()[:2]
            online_im = cv2.resize(online_im, (mo_wid//4, mo_hei//2), interpolation = cv2.INTER_AREA)
            # import ipdb; ipdb.set_trace()
            if args.show:
                cv2.imshow('Video', online_im)
                # if cv2.waitKey(25) & 0xFF == ord('q'):
                #     break
                
            # if args.save_result:
            #     vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1
    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)

    # init_model_detect_items 
    # model_items = YOLO(Yolov8Config.weight_path)
    model_items = load_model(Yolov7Config.weight_path, device=Yolov7Config.device)
    items_predictor = Yolov7Detector(model=model_items, device=Yolov8Config.device)

    pose_model = init_pose_model(PoseConfig.config, PoseConfig.checkpoint, device=PoseConfig.device.lower())
    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        logger.warning(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # init model pose
    current_time = time.localtime()
    # if args.demo == "image":
    #     image_demo(predictor, vis_folder, current_time, args)
    # elif args.demo == "video" or args.demo == "webcam":
    imageflow_demo(predictor, items_predictor, pose_model, dataset, dataset_info, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
