import sys
sys.path.insert(0, './yolov7')
import cfg
import torch
from helpers import load_model, bbox_to_center, kpts_to_box, coco_to_yolo
import glob
import scipy.io
import os
import cv2
import numpy as np
from yolov7.detect_inf import obj_detector, obj_detector_image

def convert_to_yolo(x1, y1, x2, y2, org_w, org_h):
    dw = 1./org_w
    dh = 1./org_h
    x = (x1 + x2)/2.0*dw
    y = (y1 + y2)/2.0*dh
    w = (x2 - x1)*dw
    h = (y2 - y1)*dh
    return x, y, w, h

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model("./yolov7/trained_models/best.pt", device)

cap = cv2.VideoCapture("./videos/demo_2702_v1.avi")

i = 0
if (cap.isOpened() == False):
    print("Error reading cap file")

while(cap.isOpened()):
    
    ret, frame = cap.read()
    if ret == False:
        break
    frame = cv2.resize(frame, (1920, 1080))
    org_h, org_w = frame.shape[:2]
    boxes, _ = obj_detector_image(model, frame, classes=None, device=device)
    boxes = boxes.detach().cpu().numpy()
    if i >= 100:
        if i % 10 == 0:
            with open(f"/media/ai/DATA/Hoang/ai-retail/data_record/2702/v1/labels/frame_{i}.txt", "w") as f:
                for box in boxes:
                    x1, y1, x2, y2 = box[:4]
                    cls = box[-1]
                    x, y, w, h = convert_to_yolo(x1, y1, x2, y2, org_w, org_h)
                    f.writelines(cfg.classes[int(cls)] + " "  + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")
            
            cv2.imwrite(f"/media/ai/DATA/Hoang/ai-retail/data_record/2702/v1/images/frame_{i}.jpg", frame)    
        
    i+=1

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
            break
        
cap.release()
cv2.destroyAllWindows()


    