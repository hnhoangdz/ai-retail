import sys
sys.path.insert(0, './detection/yolov7')

from typing import List, Dict
import logging
import torch
import numpy as np
import cv2
import os
import warnings
from argparse import ArgumentParser
import time
import mmcv

from tracking.mmpose.mmpose.apis import (collect_multi_frames, get_track_id,
                         inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_tracking_result)
from mmpose.core import Smoother
from mmpose.datasets import DatasetInfo

from config.config_common import CLASSES, COLOR, objects
from ultils.object import Human, Hand, Item, Box, Point, Image
from config.config_detection import YoloConfig
from ultils.preprocess import Yolov7Detector, load_model
from ultils.logger import set_logger
from config.config_detection import PoseConfig
level = logging.DEBUG
logger = set_logger(level=level)


def object_to_box(dets:List[Human]) -> tuple:
    results = []
    for each_det in dets:
        conved = {'bbox':[each_det.box.tl.x, each_det.box.tl.y, each_det.box.br.x, each_det.box.br.y, each_det.conf]}
        results.append(conved)
    return results


def make_parser():
    """Visualize the demo images.
    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--euro',
        action='store_true',
        help='(Deprecated, please use --smooth and --smooth-filter-cfg) '
        'Using One_Euro_Filter for smoothing.')
    parser.add_argument(
        '--smooth',
        action='store_true',
        help='Apply a temporal filter to smooth the pose estimation results. '
        'See also --smooth-filter-cfg.')
    parser.add_argument(
        '--smooth-filter-cfg',
        type=str,
        default='configs/_base_/filters/one_euro.py',
        help='Config file of the filter to smooth the pose estimation '
        'results. See also --smooth.')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    parser.add_argument(
        '--use-multi-frames',
        action='store_true',
        default=False,
        help='whether to use multi frames for inference in the pose'
        'estimation stage. Default: False.')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the pose'
        'estimation stage. Default: False.')
    return parser

def tracking():
    args = make_parser().parse_args()

    assert args.show or (args.out_video_root != '')

    logger.info('Initializing detection model Yolov7 and Pose model...')
    model_detector = load_model(weight_path=YoloConfig.weight_path, device=args.device.lower())
    det_model = Yolov7Detector(model=model_detector,device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        logger.warning(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # read video
    video = mmcv.VideoReader(args.video_path)
    assert video.opened, f'Faild to load video file {args.video_path}'

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = video.fps*2.5
        size = (video.width, video.height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # frame index offsets for inference, used in multi-frame inference setting
    if args.use_multi_frames:
        assert 'frame_indices_test' in pose_model.cfg.data.test.data_cfg
        indices = pose_model.cfg.data.test.data_cfg['frame_indices_test']
    # build pose smoother for temporal refinement
    if args.euro:
        warnings.warn(
            'Argument --euro will be deprecated in the future. '
            'Please use --smooth to enable temporal smoothing, and '
            '--smooth-filter-cfg to set the filter config.',
            DeprecationWarning)
        smoother = Smoother(
            filter_cfg='tracking/mmpose/configs/_base_/filters/one_euro.py', keypoint_dim=2)
    elif args.smooth:
        smoother = Smoother(filter_cfg=args.smooth_filter_cfg, keypoint_dim=2)
    else:
        smoother = None
    # whether to return heatmap, optional
    return_heatmap = False
    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    next_id = 0
    pose_results = []
    print('Running inference...')
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        start_time = time.time()
        pose_results_last = pose_results

        # get the detection results of current frame
        # the resulting box is (x1, y1, x2, y2)
        result_dets, fps_det = det_model.detect(cur_frame)
        humans, hands, items = [], [], []
        for det in result_dets:
            if det.id_object == CLASSES.hand:
                hands.append(det)
            elif det.id_object == CLASSES.person:
                humans.append(det)
            else:
                items.append(det)
        humans_bbox = object_to_box(humans)

        # keep the person class bounding boxes.
        if args.use_multi_frames:
            frames = collect_multi_frames(video, frame_id, indices,
                                          args.online)
        # test a single image, with a list of bboxes.
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            frames if args.use_multi_frames else cur_frame,
            humans_bbox,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # get track id for each person instance
        pose_results, next_id = get_track_id(
            pose_results,
            pose_results_last,
            next_id,
            use_oks=args.use_oks_tracking,
            tracking_thr=args.tracking_thr)

        # post-process the pose results with smoother
        if smoother:
            pose_results = smoother.smooth(pose_results)
        end_time = time.time()
        logger.debug(f"FPS:{round(1/(end_time-start_time), 3)}")
     








        # show the results
        vis_frame = vis_pose_tracking_result(
            pose_model,
            cur_frame,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=False)
        if args.show:
            cv2.imshow('Frame', vis_frame)

        if save_out_video:
            videoWriter.write(vis_frame)

        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if save_out_video:
        videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()



if __name__ == "__main__":
    tracking()

    
    # image = cv2.imread('/home/ubuntu/Pictures/download.jpeg')
    # humans, hands, items = [], [], []
    # for det in result_dets:
    #     if det.id_object == CLASSES.hand:
    #         hands.append(det)
    #     elif det.id_object == CLASSES.person:
    #         humans.appens(det)
    #     else:
    #         items.append(det)
    
 


