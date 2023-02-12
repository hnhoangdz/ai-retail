import sys
sys.path.insert(0, './yolov7')

import torch
from utils.helpers import load_model, bbox_to_center, kpts_to_box, coco_to_yolo
import glob
import scipy.io
import os
import cv2
import numpy as np
from yolov7.detect_inf import obj_detector

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

def hands_box(img_path, label_path):
    
    img = cv2.imread(img_path)
    org_h,org_w,_ = img.shape
    mat_label = scipy.io.loadmat(label_path)
    bboxes = []
    bn = os.path.basename(label_path).replace(".mat", "")
    
    for x in mat_label["boxes"][0]:
        box = x[0]
        tmp_box = []

        if len(box[0]) == 6:
            tl, tr, br, bl, _, _ = box[0]
        elif len(box[0]) == 4:
            tl, tr, br, bl = box[0]
            
        y1, x1 = int(tl[0][0]), int(tl[0][1])
        tmp_box.append([x1, y1])
        y2, x2 = int(tr[0][0]), int(tr[0][1])
        tmp_box.append([x2, y2])
        y3, x3 = int(br[0][0]), int(br[0][1])
        tmp_box.append([x3, y3])
        y4, x4 = int(bl[0][0]), int(bl[0][1])
        tmp_box.append([x4, y4])
        
        new_box = kpts_to_box(tmp_box)
        xmin, ymin, xmax, ymax = new_box
        x,y,w,h = coco_to_yolo(xmin, ymin, xmax, ymax, org_w, org_h)
        bboxes.append([x,y,w,h])

    return np.array(bboxes)    

def people_box(img_path, model):
    
    img = cv2.imread(img_path)
    org_h,org_w,_ = img.shape
    
    bboxes = []
    per_dets, fps_per = obj_detector(model, img)
    for box in per_dets[:,:4].numpy():
        x1, y1, x2, y2 = box
        x,y,w,h = coco_to_yolo(x1, y1, x2, y2, org_w, org_h)
        bboxes.append([x,y,w,h])
    
    return bboxes

def write_yolo_label(id, box, target_path):
    
    with open(target_path, "w") as f:
        for b in box:
            x,y,w,h = b
            f.write(str(id) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")
        f.close()
        
if __name__ == "__main__":
    root = "/home/hoangdinhhuy/hoangdinhhuy/VTI/retail_store/datasets/org_hand_dataset"
    phases = ["train", "valid", "test"] 
    
    for phase in phases:

        parent_path = os.path.join(root, phase) # train
        data_path = os.path.join(parent_path, "data") # train/data
        labels_path = os.path.join(data_path, "labels") # train/data/labels
        make_if_not_exist(labels_path)
        images_path = glob.glob(os.path.join(data_path, "images", "*.jpg"))
        
        for img_path in images_path:
            mat_path = img_path.replace("images", "annotations").replace("jpg", "mat")
            base_name = os.path.basename(img_path).replace("jpg", "txt")
            bn = base_name.replace(".txt", "")
            hands_bboxes = hands_box(img_path, mat_path)

            write_yolo_label(0, hands_bboxes, os.path.join(labels_path, base_name))




