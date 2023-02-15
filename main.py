
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

class Behavior:
    # Identify behavior of one person during consecutive 10 frames
    def __init__(self, humans, items, current_frame) -> None:
        self.humans = humans
        self.items = items
        self.current_frame = current_frame
        
    def get_item(self) -> bool:
        
        if len(self.humans) < 1: return False
        
        area = ""
        is_hand_picking = False
        
        for human in self.humans:
            
            print(human.__class__.__name__)
            human_box = human.box
            midx, midy = human_box.center_point()
            area = search_area(self.current_frame, midx, midy)
            print(area)
            if area != "shelf":
                return False
            
            human_hands = human.hands
            
            for hand in human_hands:
                print(hand.__class__.__name__)
                for item in self.items:
                    print(item.__class__.__name__)
                    is_hand_picking = hand.touch(item)
                    
        return True if (area == "shelf" and is_hand_picking) else False
    
    def put_item_to_shelf(self) -> bool:
        pass

    def bring_item_to_pay(self) -> bool:
        pass

    def stolen(self) -> bool:
        '''
           :))
        '''
        pass


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
        
        total_frame_humans = []
        total_frame_items = []
        
        while True:
            
            start_time = time.time()
            ret, frame = self.video.read()
            frame_copy = frame.copy()
            if not ret:
                print("error when reading camera")
                break
            
            org_h, org_w = self.im_height, self.im_width
            print(org_h, org_w)
            # Frame Processing
            frame_process = Frame(ith, frame, detector, tracker, display_area=True)
            frame_process.detect(args.conf_thresh, args.iou_thresh, device)
            frame_process.tracking(args.input_size, args.aspect_ratio_thresh, args.min_box_area)
            
            # Get all objects apprearing in current frame
            frame_objs = frame_process.frame_objs
            
            # for obj in frame_objs:
            #     if obj.cls_id == 0:
            #         human_box = obj.box
            #         top,left = human_box.top_left.x, human_box.top_left.y
            #         bottom, right =  human_box.bot_right.x, human_box.bot_right.y
            #         print(top, left, bottom,right)
            #         human_hands_box = obj.hands
                
            # Append all objects to list in current frame
            total_frame_humans.append(frame_objs["humans"])
            total_frame_items.append(frame_objs["items"])
            
            # 10 consecutive frames
            if len(total_frame_humans) == 11:
                total_frame_humans.pop(0)
                total_frame_items.pop(0)
            
            if ith >= 10:
                for frame_humans in total_frame_humans:
                    behavior = Behavior(frame_humans, total_frame_items, frame_copy)
                    print(behavior.get_item())
                    
            # Display FPS
            end_time = time.time()
            fps = 1/(end_time - start_time)
            cv2.putText(frame, "FPS : {}".format(int(fps)), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
            ith +=1             
            
            if self.args.display:
                cv2.namedWindow("Retail", cv2.WINDOW_KEEPRATIO)
                cv2.imshow("", frame)
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




