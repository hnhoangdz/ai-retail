import os
import shutil
import glob

import cv2
import time
import requests
import random
import numpy as np
from PIL import Image
from pathlib import Path
from collections import OrderedDict, namedtuple
import torch
import os
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import time_synchronized
import sys

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):

    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def obj_detector(detector, image, conf_thresh=0.15, iou_thresh=0.35):
    
    half = False
    img = image.copy()
    
    img = letterbox(img, new_shape=(640, 640))[0]
    
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to("cpu")
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        pred = detector(img, augment=False)[0]  # list: bz * [ (#obj, 6)]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=0, agnostic=False)
    results_det = pred[0]
    if results_det is not None and len(results_det):
        results_det[:, :4] = scale_coords(img.shape[2:], results_det[:, :4], image.shape).round()
        # for res in results_det:
        #     x1,y1,x2,y2,conf,cls = res
        #     cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0),1,cv2.LINE_AA)
        #     cv2.putText(image, "{:.2f}".format(conf.item()), (int(x1),int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        # cv2.imshow("", image)
        # cv2.waitKey(0)
        return results_det
    else:
        return None
    
def convert_to_yolo(x1, y1, x2, y2, org_w, org_h):
    dw = 1./org_w
    dh = 1./org_h
    x = (x1 + x2)/2.0*dw
    y = (y1 + y2)/2.0*dh
    w = (x2 - x1)*dw
    h = (y2 - y1)*dh
    return x, y, w, h

if __name__ == "__main__":
    model = torch.load("/home/hoangdinhhuy/hoangdinhhuy/VTI/725/ai-digital-signage/yolov7/yolov7.pt", map_location="cpu")["model"].float().to("cpu").eval()
    for img_path in glob.glob(os.path.join("/home/hoangdinhhuy/hoangdinhhuy/VTI/725/youtube_dataset/images", "*.jpg")):
        
        image = cv2.imread(img_path)
        org_h, org_w, _ = image.shape
        pred = obj_detector(model, image)
        label_path = img_path.replace("images", "labels").replace("jpg", "txt")
        with open(label_path, "w") as f:
            for i, res in enumerate(pred):
                x1,y1,x2,y2,conf,cls = res.cpu().detach().numpy()
                x,y,w,h = convert_to_yolo(x1,y1,x2,y2,org_w,org_h)
                if i != len(pred) - 1:
                    f.writelines("0"+ " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")
                else:
                    f.writelines("0"+ " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h))
