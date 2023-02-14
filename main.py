
import sys
sys.path.insert(0, './yolov7')
import torch
from torchvision import transforms

from yolov7.detect_inf import obj_detector
from yolov7.keypoint_inf import get_keypoints, get_hands_kpts, get_hands_box
from yolov7.utils.torch_utils import select_device

from byte_track.tracker.byte_tracker import BYTETracker
from byte_track.tracking_utils.timer import Timer

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time

from helpers import draw_box, draw_line, load_model, bbox_to_center, search_area
from cfg import *
from frame_process import Frame

class VideoRetailStore(object):
    def __init__(self, args):
        self.args = args
        self.device = select_device(args.device) 
        self.half = self.device.type != 'cpu'    
        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.video = cv2.VideoCapture(args.cam)
        else:
            self.video = cv2.VideoCapture()

    def __enter__(self):
        if self.args.cam != "-1":
            print('Camera ...')
            ret, frame = self.video.read()
            assert ret, "Error: Camera error"
            self.im_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ************************* Load video from file *************************
        else:
            assert os.path.isfile(self.args.input_path), "Path error"
            self.video.open(self.args.input_path)
            self.im_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.video.isOpened()
            print('Done. Load video file ', self.args.input_path)

        # ************************* create output *************************
        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)
            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, os.path.basename(self.args.input_path))

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*self.args.fourcc)
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc,
                                          self.video.get(cv2.CAP_PROP_FPS), (self.im_width, self.im_height))
            print('Done. Create output file ', self.save_video_path)

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.video.release()
        self.writer.release()
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
            self.video = cv2.VideoCapture()
            
    def run(self):
        ith = 0
        
        while True:
            
            start_time = time.time()
            ret, frame = self.video.read()
            if not ret:
                print("error when reading camera")
                break
            
            org_h, org_w = frame.shape[:2]
            
            frame_obj = Frame(ith, frame, detector, tracker, display_area=True)
            frame_obj.detect(args.conf_thresh, args.iou_thresh, device)
            frame_obj.tracking(args.input_size, args.aspect_ratio_thresh, args.min_box_area)
            print(len(frame_obj.frame_objs))
            for box in frame_obj.frame_objs:
                print(box.cls_id)
            # Return Objects
            # human_boxes, hand_boxes, item_boxes = frame_obj.human_boxes, frame_obj.hand_boxes, frame_obj.item_boxes

            # if len(human_boxes) >= 1:
            #     p_track_boxes = frame_obj.tracking(human_boxes, args.input_size, 
            #                                        args.aspect_ratio_thresh, 
            #                                        args.min_box_area) # [x1,y1,x2,y2,id]
            #     for p_box in p_track_boxes:
                    
            
            # area = search_area(frame, 998, 225)
            # print(area)
            
            # # Object detection
            # obj_dets, _ = obj_detector(person_detector, frame)
            # obj_dets = obj_dets.detach().cpu().numpy()
            # per_dets = obj_dets[obj_dets[:, 5] == 0]
            # box = per_dets[0]
            # # x1, y1, x2, y2 = box
            # cx, cy = bbox_to_center(box[:4])
            # search_area(frame, 788, 552)
            
            # # Objects tracking
            # online_targets = tracker.update(per_dets, frame.shape[:2], self.args.input_size)
            # input_h, input_w = self.args.input_size
            # scale = min(input_h / float(org_h), input_w / float(org_w))
            # online_tlwhs = []
            # online_ids = []
            # online_scores = []
            
            # # check person is invalid or not
            # # it must be: w < h & area > 10
            # for t in online_targets:
            #     tlwh = t.tlwh
            #     tid = t.track_id
            #     vertical = tlwh[2] / tlwh[3] > self.args.aspect_ratio_thresh
            #     if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
            #         online_tlwhs.append(tlwh)
            #         online_ids.append(tid)
            #         online_scores.append(t.score)
                
            # for idx, tlwh in enumerate(online_tlwhs):
            #     tlwh = tlwh * scale
            #     cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3]) ), (255, 0, 0), 2)
            #     cv2.putText(frame, str(online_ids[idx]), (int(tlwh[0]), int(tlwh[1] - 7)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                # cv2.putText(frame,  "{:.2f}".format(online_scores[idx]), (int(tlwh[2])-20, int(tlwh[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            
            # for i, box in enumerate(obj_dets[:,:4].numpy()):
            #     x1, y1, x2, y2 = box
            #     # cx, cy = bbox_to_center(box)
            #     # cv2.circle(frame, (cx, cy), 2, (125, 0, 255), 2, cv2.LINE_AA)
            #     # area = search_area(frame, cx, cy)
            #     # print(area)
            #     # color = (255, 0, 0)
            #     # if area == "shelf":
            #     #     color = (0, 0, 255)
            #     # cv2.putText(frame, area, (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
            #     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
            # Item detection
            # item_dets, fps_item = obj_detector(item_detector, frame)
            # for x1,y1,x2,y2 in item_dets[:,:4]:
            #     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                
            # Body Keypoints
            # body_kpts = get_keypoints(frame, obj_dets)
            # if len(body_kpts) > 0:
            #     hand_kpts = get_hands_kpts(body_kpts)
            #     hand_box = get_hands_box(hand_kpts)
            #     print(hand_box)
            
            cv2.imwrite("c.jpg", frame)
            end_time = time.time()
            fps = 1/(end_time - start_time)
            cv2.putText(frame, "FPS : {}".format(int(fps)), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
            ith +=1 
            if self.args.display:
                cv2.namedWindow("test", cv2.WINDOW_KEEPRATIO)
                cv2.imshow("test", frame)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    cv2.destroyAllWindows()
                    break
                
            if self.args.save_path:
                self.writer.write(frame)
            
def get_parser():
    parser = argparse.ArgumentParser("Retail Store Demo!")
    
    # Detection
    parser.add_argument("--weights_path", type=str, default="./yolov7/trained_models/best_record_v3.pt")
    # parser.add_argument("--weights_item", type=str, default="/home/hoangdinhhuy/hoangdinhhuy/VTI/retail_store/yolov7/trained_models/item_detector.pt")
    parser.add_argument("--conf_thresh", type=float, default=0.45, help="confidence threshold object detection")
    parser.add_argument("--iou_thresh", type=float, default=0.3, help="iou threshold object detection")
    parser.add_argument("--video", type=str, default="", help="input video link")

    # Input and ouput
    parser.add_argument("--input_path", type=str, default="/home/hoangdinhhuy/hoangdinhhuy/VTI/retail_store/output.avi", help="input video")
    parser.add_argument("--save_path", type=str, default="/home/hoangdinhhuy/hoangdinhhuy/VTI/retail_store/results", help="output folder")
    parser.add_argument("--fourcc", type=str, default="mp4v", help="output video codec (verify ffmpeg support)")
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # Camera
    parser.add_argument("--display", default=True, action="store_true")
    parser.add_argument("--cam", action="store", type=str, default="-1")

    # Tracking args
    parser.add_argument("--input_size", type=tuple, default=(800, 1440), help="input size image in tracker")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.2,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    args = get_parser().parse_args()
    
    # Detection
    detector = load_model(args.weights_path, device)
    # item_detector = load_model(args.weights_item, device)
        
    # Tracking
    tracker = BYTETracker(args)
    with VideoRetailStore(args) as vidRS:
        vidRS.run()




