
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

class Behavior:
    
    # Identify behavior of people during consecutive 10 frames
    def __init__(self, consecutive_humans, consecutive_items, last_human_ids) -> None:
        self.consecutive_humans = consecutive_humans # i -> i+10
        self.consecutive_items = consecutive_items # i -> i+10
        self.last_human_ids = last_human_ids
        
    def get_item(self):
        
        status = {}
        
        # Iterate all humans id
        for id in self.last_human_ids:
            
            # Check last frame
            for human_obj in self.consecutive_humans[-1]:
                
                status[human_obj] = {
                    "area": "",
                    "items": []
                }
                
                if human_obj.id == id:
                    status[human_obj]["is_payed"] = False
                    human_box = human_obj.box
                    midx, midy = human_box.center_point()
                    area = search_area(1920, 1080, midx, midy)
                    status[human_obj]["area"] = area
                    
                    in_area = True if area != "outside" else False
                    
                    if in_area == False:
                        continue
                    
                    human_hands = human_obj.hands
                    
                    for hand_obj in human_hands:
                        for item_obj in self.consecutive_items[-1]:
                            if hand_obj.touch(item_obj) and in_area:
                                if item_obj not in status[human_obj]:
                                    status[human_obj]["items"].append(item_obj)
                                    
            in_area = True
            is_touching = True
            curr_obj = None
            
            # Itearte each frame in remain
            for frame_humans, frame_items in zip(self.consecutive_humans[:-1], 
                                                 self.consecutive_items[:-1]):
                
                # Iterate humans object in each frame
                for human_obj in frame_humans:
                    
                    if human_obj.id == id:
                        curr_obj = human_obj
                        human_box = human_obj.box
                        midx, midy = human_box.center_point()
                        area = search_area(1920, 1080, midx, midy)
                        in_area = True if area != "outside" else False
                        human_hands = human_obj.hands
                        
                        for hand_obj in human_hands:
                            for item_obj in frame_items:
                                
                                if hand_obj.touch(item_obj):
                                    is_touching = True
                       
            confirm = in_area and is_touching
            if confirm == False:
                status[curr_obj]["items"] = []
                        
        return status
    
    def put_item_to_shelf(self):
        pass

    def bring_item_to_pay(self, current_state, items_on_shelf):
                
        for human in current_state:
            on_store = False
            if current_state[human]["area"] == "payment":
                items_obj = current_state[human]["items"]
                for item in items_obj:
                    on_store = item.cls_id in items_on_shelf
            current_state[human]["is_payed"] = on_store
        
        return current_state
            

    def stolen(self):
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
                if len(consecutive_frame_humans) == 11:
                    consecutive_frame_humans.pop(0)
                    consecutive_frame_items.pop(0)
                
                # Get behavior of human
                if ith >= 10:
                    behavior = Behavior(consecutive_frame_humans, consecutive_frame_items, last_human_ids)
                    current_state = behavior.get_item()
                    current_state = behavior.bring_item_to_pay(current_state, items_on_shelf)
                    print("aaaaaaaaaaaa ", current_state)
                    print("=============================================")
                    current_state = behavior.bring_item_to_pay(current_state, items_on_shelf)
                    # print(current_state)
                    # visualization
                    cv2.putText(frame, "FPS : {}".format(int(fps)), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, COLOR.green)
                    padding_text = 6
                    for human, meta_data in current_state.items():
                        area = meta_data['area']
                        items = meta_data['items']
                        payed = meta_data['is_payed']
                        color_human = COLOR.green if area == 'payment' else COLOR.blue
                        visualize_human(human, 
                            image=frame, 
                            color=color_human,
                            thickness=2,
                            label=f"{classes[human.cls_id]}: {human.id}",
                        )




                        color_item = COLOR.yellow if payed==True else COLOR.magenta
                        for item in items:
                            visualize_item(
                                item,
                                frame,
                                color=color_item,
                                thickness=2,
                                label=classes[item.cls_id]

                            )
                    
                    
                    

                            


                    # import ipdb; ipdb.set_trace()

                    # visual_object(input=current_state, image=frame)
                # print(current_state)
                # print("=============================================")
            
            
                
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
                
            # if self.args.save_path:
            #     self.writer.write(frame)
                                
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
    parser.add_argument("--track_thresh", type=float, default=0.45, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.6, help="matching threshold for tracking")
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




