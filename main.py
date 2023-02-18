
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
from ultils.visualize import visualize_human, visualize_item, visual_object
from behavior import ByteTrackBehavior




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
            # print(exc_type, exc_value, exc_traceback)
            self.video = cv2.VideoCapture()
            
    def run(self):
        
        ith = 0
        
        consecutive_frame_humans = []
        consecutive_frame_items = []
        # previous_states = []
        
        items_on_shelf = None
        last_human_ids = None
        while True:
            
            start_time = time.time()
            ret, frame = self.video.read()
            if not ret:
                print("error when reading camera")
                break

            # Init frame object 
            frame_process = Frame(ith, frame, detector, tracker, display_area=True)
            
            # Frame detection
            frame_process.detect(args.conf_thresh, args.iou_thresh, device)
            
            # If there are detected objects 
            if frame_process.is_detected:

                # Return id of items that appeared first on shelf
                if items_on_shelf is None:
                    items_on_shelf = frame_process.get_items_on_shelf()
                
                # Tracking humans
                frame_process.tracking(args.input_size, args.aspect_ratio_thresh, args.min_box_area)
                
                # Get all objects apprearing in current frame
                # Return a dictionary, contains humans & items object
                curr_frame_objs = frame_process.frame_objs 
                last_human_ids = [human_obj.id for human_obj in curr_frame_objs["humans"]]
                
                # Append all objects to list in current frame
                consecutive_frame_humans.append(curr_frame_objs["humans"])
                consecutive_frame_items.append(curr_frame_objs["items"])
                
                # 10 consecutive frames
                if len(consecutive_frame_humans) == 2:
                    consecutive_frame_humans.pop(0)
                    consecutive_frame_items.pop(0)
                
                # Get behavior of human
                if ith >= 10:
                    behavior = ByteTrackBehavior(consecutive_frame_humans, consecutive_frame_items, last_human_ids)
                    # print("last_human_ids ", last_human_ids)
                    current_state = behavior.get_item()
                    current_state = behavior.bring_item_to_pay(current_state, items_on_shelf)
                    print("aaaaaaaaaaaa ", current_state)
                    print("=============================================")
                    current_state = behavior.bring_item_to_pay(current_state, items_on_shelf)
                    # print(current_state)

                    # visualization
                    x_text_start = 5
                    y_text_start = 130
                    for human, meta_data in current_state.items():
                        area = meta_data['area']
                        items = meta_data['items']
                        payed = meta_data['is_payed']
                        color_human = COLOR.green if area == "payment" else COLOR.blue
                        visualize_human(human, 
                            image=frame, 
                            color=color_human,
                            thickness=2,
                            label=f"{classes[human.cls_id]}: {human.id}",
                        )
                        cv2.putText(frame, f"person {human.id}:", (x_text_start, y_text_start), cv2.FONT_HERSHEY_COMPLEX, 0.7, COLOR.green)
                        x_start_item = x_text_start + 120

                        color_item = COLOR.yellow if payed==True else COLOR.magenta
                        if area == "payment":
                            color_item == COLOR.yellow
                        elif area == "attend":
                            color_item == COLOR.magenta
                        else:
                            continue

                        for item in items:
                            visualize_item(
                                item,
                                frame,
                                color=color_item,
                                thickness=2,
                                label=classes[item.cls_id]
                            )
                            cv2.putText(frame, f"{classes[item.cls_id]}, ", (x_start_item, y_text_start), cv2.FONT_HERSHEY_COMPLEX, 0.7, COLOR.magenta)
                            x_start_item += 80
                        y_text_start += 20

                    # # visual text
                    # x_text_start = 5
                    # y_text_start = 130
                    # for human, meta_data in current_state.items():
                    #     cv2.putText(frame, f"person {human.id}:", (x_text_start, y_text_start), cv2.FONT_HERSHEY_COMPLEX, 0.7, COLOR.green)
                    #     x_start_item = x_text_start + 120
                    #     for item in items:
                    #         cv2.putText(frame, f"{classes[item.cls_id]}, ", (x_start_item, y_text_start), cv2.FONT_HERSHEY_COMPLEX, 0.7, COLOR.magenta)
                    #         x_start_item += 80
                    #     y_text_start += 20


                
            else:
                consecutive_frame_humans = []
                consecutive_frame_items = []
                                
            # Display FPS

            end_time = time.time()
            fps = 1/(end_time - start_time)
            cv2.putText(frame, "FPS : {}".format(int(fps)), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, COLOR.green)
            ith += 1      
                   
            cv2.imwrite("visualization_image.jpg", frame)
            
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
    parser.add_argument("--weights_path", type=str, default="./weights/best.pt")
    # parser.add_argument("--weights_item", type=str, default="/home/hoangdinhhuy/hoangdinhhuy/VTI/retail_store/yolov7/trained_models/item_detector.pt")
    parser.add_argument("--conf_thresh", type=float, default=0.45, help="confidence threshold object detection")
    parser.add_argument("--iou_thresh", type=float, default=0.3, help="iou threshold object detection")
    parser.add_argument("--video", type=str, default="", help="input video link")

    # Input and ouput
    parser.add_argument("--input_path", type=str, default="/home/hoangdinhhuy/hoangdinhhuy/VTI/retail_store/output.avi", help="input video")
    parser.add_argument("--save_path", type=str, default="./results", help="output folder")
    parser.add_argument("--fourcc", type=str, default="mp4v", help="output video codec (verify ffmpeg support)")
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # Camera
    parser.add_argument("--display", default=True, action="store_true")
    parser.add_argument("--cam", action="store", type=str, default="-1")

    # Tracking args
    parser.add_argument("--input_size", type=tuple, default=(800, 1440), help="input size image in tracker")
    parser.add_argument("--track_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.45, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.0,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Devide:{device}")
    args = get_parser().parse_args()
    
    # Detection
    detector = load_model(args.weights_path, device)
    # item_detector = load_model(args.weights_item, device)
        
    # Tracking
    tracker = BYTETracker(args)
    with VideoRetailStore(args) as vidRS:
        vidRS.run()




