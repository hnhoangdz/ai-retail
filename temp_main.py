from behavior import ByteTrackBehavior
from ultils.visualize import visualize_human, visualize_item, visual_object, visualize_payment
from frame_process import Frame
from cfg import *
from helpers import draw_box, draw_line, load_model, bbox_to_center, search_area
import time
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
from byte_track.tracking_utils.timer import Timer
from byte_track.tracker.byte_tracker import BYTETracker
from yolov7.utils.torch_utils import select_device
from yolov7.keypoint_inf import get_keypoints, get_hands_kpts, get_hands_box
from yolov7.detect_inf import obj_detector
from torchvision import transforms
import torch
import sys

sys.path.insert(0, './yolov7')
from cfg import *

in_frame = None

def read_frame(video):
    global in_frame

    while True:
        time.sleep(0.02)
        ret, frame = video.read()
        if ret:
            in_frame = frame
        else:
            time.sleep(1)

class VideoRetailStore(object):
    def __init__(self, args):
        self.args = args
        self.device = select_device(args.device)
        self.half = self.device.type != 'cpu'
        # Detection
        self.human_detector = load_model(args.weights_human, device)
        self.hands_items_detector = load_model(args.weights_hands_items, device)
        # Tracking
        self.tracker = BYTETracker(args)
        
        # 
        # self.consecutive_frame_humans = []
        # self.consecutive_frame_items = []
        # self.attend_storage = {}
        # self.payment_storage = {}
        # self.payment_frame = np.zeros([1080, 1920//4, 3], dtype=np.uint8)
        # self.payment_frame[:, :] = [255, 255, 255]

    def __enter__(self):
        # if self.args.cam != "-1":
        #     print('Camera ...')
        #     ret, frame = self.video.read()
        #     assert ret, "Error: Camera error"
        #     self.im_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     self.im_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # # ************************* Load video from file *************************
        # else:
        #     assert os.path.isfile(self.args.input_path), "Path error"
        #     self.video.open(self.args.input_path)
        #     self.im_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     self.im_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #     assert self.video.isOpened()
        #     print('Done. Load video file ', self.args.input_path)

        # # ************************* create output *************************
        # if self.args.save_path:
        #     os.makedirs(self.args.save_path, exist_ok=True)
        #     # path of saved video and results
        #     self.save_video_path = os.path.join(
        #         self.args.save_path, os.path.basename(self.args.input_path))
        #     # create video writer
        #     fourcc = cv2.VideoWriter_fourcc(*self.args.fourcc)
        #     self.writer = cv2.VideoWriter(self.save_video_path, fourcc,
        #                                   self.video.get(cv2.CAP_PROP_FPS), (self.im_width, self.im_height))
        #     print('Done. Create output file ', self.save_video_path)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # self.video.release()
        # self.writer.release()
        # if exc_type:
        #     # print(exc_type, exc_value, exc_traceback)
        #     self.video = cv2.VideoCapture()
        pass

    def run(self):
        ith = 0
        consecutive_frame_humans = []
        consecutive_frame_items = []
        attend_storage = {}
        payment_storage = {}
        payment_frame = np.zeros([1080, 1920//4, 3], dtype=np.uint8)
        payment_frame[:, :] = [255, 255, 255]

        while True:
            start_time = time.time()
            frame = in_frame.copy()
     
            print("ith ", ith)
            frame = cv2.resize(frame, (1920, 1080))
            # Init frame object
            frame_process = Frame(ith, frame, self.human_detector, self.hands_items_detector,
                                    self.tracker, display_area=True)
            # Frame detection
            frame_process.detect(args.conf_thresh, args.iou_thresh, device)

            # If there are detected objects
            if frame_process.is_detected:

                # Tracking humans
                frame_process.tracking(args.input_size, args.aspect_ratio_thresh, args.min_box_area)

                # Get all objects apprearing in current frame
                # Return a dictionary, contains humans & items object
                curr_frame_objs = frame_process.frame_objs

                # Append all objects to list in current frame
                consecutive_frame_humans.append(curr_frame_objs["humans"])
                consecutive_frame_items.append(curr_frame_objs["items"])

                # 10 consecutive frames
                if len(consecutive_frame_humans) == (CONSECUTIVE_FRAME + 1):
                    consecutive_frame_humans.pop(0)
                    consecutive_frame_items.pop(0)

                # Get behavior of human
                if ith >= 10:
                    behavior = ByteTrackBehavior(consecutive_frame_humans, consecutive_frame_items)
                    # print("last_human_ids ", last_human_ids)
                    current_state = behavior.get_item()
                    # print(" current_state ", current_state)
                    # Attend storage
                    for current_state_key, current_state_values in current_state.items():
                        
                        human_id = current_state_key.id
                        area = current_state_values["area"]
                        items = current_state_values["items"]
                        
                        if (area == "attend"):
                            if (len(items) > 0):
                                if human_id not in attend_storage:
                                    attend_storage[human_id] = items
                                else:
                                    attend_items = [i.cls_id for i in attend_storage[human_id]]
                                    for item_obj in items:
                                        if item_obj.cls_id not in attend_items:
                                            attend_storage[human_id].append(item_obj)
                                            
                    current_state, attend_storage, payment_storage = behavior.bring_item_to_pay(current_state, attend_storage, payment_storage)
                    print("=============================================")
                    # Visualization
                    x_text_start = 5
                    y_text_start = 130
                    for human, meta_data in current_state.items():
                        
                        human_id = human.id
                        area = meta_data['area']
                        items = meta_data['items']
                        
                        if area == "shelf":
                            color_human = COLOR.blue
                            color_item = COLOR.blue
                        elif area == "attend":
                            color_human = COLOR.magenta
                            color_item = COLOR.magenta
                        else:
                            continue
                        
                        cv2.putText(frame, f"person {human_id}:", (x_text_start, y_text_start), 
                                    cv2.FONT_HERSHEY_COMPLEX, 0.7, color_human)
                        
                        x_start_item = x_text_start + 120
                        
                        visualize_human(human,
                                        image=frame,
                                        color=color_human,
                                        thickness=2,
                                        label=f"person: {human.id}",
                                        area=area
                                        )
                        if area == "attend":
                            if human_id in attend_storage:
                                for item in attend_storage[human_id]:
                                    cv2.putText(frame, f"{classes[item.cls_id]}    ", 
                                                (x_start_item, y_text_start), 
                                                cv2.FONT_HERSHEY_COMPLEX, 0.7, color_item)
                                    x_start_item += 80
                        elif area == "shelf":
                            for item in items:
                                cv2.putText(frame, f"{classes[item.cls_id]}    ", 
                                            (x_start_item, y_text_start), 
                                            cv2.FONT_HERSHEY_COMPLEX, 0.7, color_item)
                                x_start_item += 80
                        y_text_start += 20

            else:
                consecutive_frame_humans = []
                consecutive_frame_items = []

            
            ith += 1
            # cv2.imwrite("visualization_image.jpg", frame)

            # Display FPS
            end_time = time.time()
            fps = 1/(end_time - start_time)
            cv2.putText(frame, "FPS : {}".format(int(fps)), (5, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, COLOR.green)
            cv2.namedWindow("Retail", cv2.WINDOW_KEEPRATIO)
            retail_frame = cv2.resize(frame, (1920*3//4, 1080))
            visualize_payment(payment_frame, payment_storage)
            final_frame = np.concatenate((retail_frame, payment_frame), 1)
            cv2.imshow("Retail", final_frame)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                cv2.destroyAllWindows()
                break

            # if self.args.save_path:
            #     self.writer.write(frame)


def get_parser():

    parser = argparse.ArgumentParser("Retail Store Demo!")

    # Detection
    parser.add_argument("--weights_human", type=str,
                        default="./yolov7/trained_models/yolov7.pt")
    parser.add_argument("--weights_hands_items", type=str, 
                        default="./yolov7/trained_models/hands_items_yolov7.pt")
    parser.add_argument("--conf_thresh", type=float, default=0.45,
                        help="confidence threshold object detection")
    parser.add_argument("--iou_thresh", type=float, default=0.3,
                        help="iou threshold object detection")
    parser.add_argument("--video", type=str, default="",
                        help="input video link")

    # Input and ouput
    parser.add_argument("--input_path", type=str,
                        default="./videos/cut_p1.avi", help="input video")
    parser.add_argument("--save_path", type=str,
                        default="./results", help="output folder")
    parser.add_argument("--fourcc", type=str, default="MJPG",
                        help="output video codec (verify ffmpeg support)")
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # Camera
    parser.add_argument("--display", default=True, action="store_true")
    parser.add_argument("--cam", action="store", type=str, default="-1")

    # Tracking args
    parser.add_argument("--input_size", type=tuple,
                        default=(800, 1440), help="input size image in tracker")
    parser.add_argument("--track_thresh", type=float,
                        default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=40,
                        help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float,
                        default=0.6, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.0,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float,
                        default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False,
                        action="store_true", help="test mot20.")
    return parser


if __name__ == "__main__":
    import threading
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Devide:{device}")
    args = get_parser().parse_args()

    if args.cam != "-1":
        video = cv2.VideoCapture(args.cam)
        print('Done. Load stream ', str(args.cam))
    else:
        assert os.path.isfile(args.input_path), "Input path error !!!"
        video = cv2.VideoCapture(args.input_path)
        print('Done. Load video file ', args.input_path)
        
    im_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        # path of saved video and results
        save_video_path = os.path.join(args.save_path, os.path.basename(args.input_path))
        # create video writer
        fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
        writer = cv2.VideoWriter(save_video_path, fourcc,
                                video.get(cv2.CAP_PROP_FPS), 
                                (im_width, im_height))
        print('Done. Create output file ', save_video_path)
    threading.Thread(target=read_frame, args=(video,)).start()
    
    with VideoRetailStore(args) as vidRS:
        vidRS.run()
