import sys
sys.path.insert(0, './yolov7')
import torch
from torchvision import transforms

from yolov7.detect_inf import obj_detector
from yolov7.utils.torch_utils import select_device

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time

from helpers import draw_box, draw_line, load_model, bbox_to_center, search_area
import glob
import shutil

names = ['dark_noodles', 'g7', 'hand', 'haohao', 'human', 'modern', 'nabaty', 'nescafe', 'oreo']

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

def coco_to_yolo(xmin, ymin, xmax, ymax, org_w, org_h):
    xcb, ycb, wb, hb = (xmax+xmin)/2, (ymax+ymin)/2, xmax-xmin, ymax-ymin
    return xcb/org_w, ycb/org_h, wb/org_w, hb/org_h

def write_labels(boxes, org_h, org_w, label_path):
    with open(label_path, "w") as f:
        for box in boxes.detach().cpu().numpy():
            id = box[-1]
            if id == 4:
                continue
            name = names[int(id.item())]
            x1,y1,x2,y2 = box[:4]
            x,y,w,h = coco_to_yolo(x1,y1,x2,y2,org_w,org_h)
            f.writelines(name + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")
        f.close()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model("/media/ai/DATA/Hoang/retail_store/yolov7/runs/train/final_retail10/weights/best.pt", device)

root_path = "/media/ai/DATA/Hoang/retail_store/final_full_data_record"
images_dir = os.listdir(root_path)

for dir in images_dir:
    full_path = os.path.join(root_path, dir)
    image_path = os.path.join(full_path, "images")
    label_path = os.path.join(full_path, "labels")
    
    make_if_not_exist(image_path)
    make_if_not_exist(label_path)
    
    images_path = glob.glob(os.path.join(full_path, "*.jpg"))
    for img_path in images_path:
        base_name = os.path.basename(img_path).replace("jpg", "txt")
        img = cv2.imread(img_path)
        org_h, org_w = img.shape[:2]
        boxes, _ = obj_detector(model, 
                                img,
                                classes=None,
                                conf_thresh=0.45, 
                                iou_thresh=0.3,
                                device=device)
        save_label_path = os.path.join(label_path, base_name)
        write_labels(boxes, org_h, org_w, save_label_path)
        shutil.move(img_path, image_path)