import sys
sys.path.insert(0, './yolov7')
import torch
from torchvision import transforms

from yolov7.detect_inf import obj_detector
# from yolov7.keypoint_inf import get_keypoints, get_hands_kpts, get_hands_box
from yolov7.utils.torch_utils import select_device

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time

from helpers import draw_box, draw_line, load_model, bbox_to_center, search_area

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hands_items_model = load_model("./yolov7/trained_models/hands_items_yolov7.pt", device)
human_model = load_model("/media/ai/DATA/Hoang/ai-retail/yolov7/trained_models/yolov7-tiny.pt", device)

cap = cv2.VideoCapture("rtsp://admin:123456aA@10.1.134.142:554/Streaming/channels/001")

if (cap.isOpened() == False):
    print("Error reading cap file")
    
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

result = cv2.VideoWriter('test_demo.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
press_detect = True
while(cap.isOpened()):
    start_time = time.time()
    
    ret, frame = cap.read()
    if ret == False:
        print(111)
        break
    t0 = time.time()
    k = cv2.waitKey(1)
    if k == ord('d'):
        press_detect = True
    elif k == ord('n'):
        press_detect =  False
    if press_detect:
        human_results_det, hand_results_det, item_results_det = obj_detector(human_model, 
                                                                                hands_items_model, 
                                                                                frame,
                                                                                classes=None,
                                                                                conf_thresh=0.45, 
                                                                                iou_thresh=0.3,
                                                                                device=device)
        for box in item_results_det:
            x1,y1,x2,y2 = box[:4]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2, cv2.LINE_AA)
        for box in human_results_det:
            x1,y1,x2,y2 = box[:4]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2, cv2.LINE_AA)
        for box in hand_results_det:
            x1,y1,x2,y2 = box[:4]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(frame, str(int(1/(time.time()-start_time))), 
                (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 
                (0,255,0), 1, cv2.LINE_AA)
    print("press detect ", press_detect)           
    frame = cv2.resize(frame, (1920, 1080))
    print("time: ", time.time() - t0)
    cv2.imshow('Frame', frame)
    if k == ord('s'):
        break
result.release()
cap.release()
cv2.destroyAllWindows()