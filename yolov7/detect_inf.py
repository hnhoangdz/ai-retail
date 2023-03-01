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
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.torch_utils import time_synchronized
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



def obj_detector(human_detector, hands_items_detector, 
                image, classes=None, conf_thresh=0.45, 
                iou_thresh=0.3, device="cpu"):

    # half = True
    # if str(device) == "cpu":
    #     half = False

    img = image.copy()
    
    img = letterbox(img, new_shape=(640, 640))[0]
    
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img =img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    img_copy = torch.clone(img)

    # Inference
    with torch.no_grad():
        human_pred = human_detector(img, augment=False)[0]  # list: bz * [ (#obj, 6)]
        hand_item_pred = hands_items_detector(img, augment=False)[0]
        
    # Apply NMS
    human_pred = non_max_suppression(human_pred, conf_thresh, iou_thresh, 0, agnostic=False)
    hand_item_pred = non_max_suppression(hand_item_pred, conf_thresh, iou_thresh, classes, agnostic=False)
    
    # Get final results
    hand_item_results_det = hand_item_pred[0]
    human_results_det = human_pred[0]

    if len(human_results_det) >= 1:
        human_results_det[:, :4] = scale_coords(img.shape[2:], human_results_det[:, :4], image.shape).round()
        human_results_det = human_results_det.detach().cpu().numpy()
    else:
        human_results_det = np.zeros([0,5])

    if len(hand_item_results_det) >= 1:
        hand_item_results_det[:, :4] = scale_coords(img.shape[2:], hand_item_results_det[:, :4], image.shape).round()
        hand_item_results_det = hand_item_results_det.detach().cpu().numpy() 
        classes_id = hand_item_results_det[:, 5]
        hand_results_det = hand_item_results_det[classes_id == 2]
        item_results_det = hand_item_results_det[classes_id != 2]
        if len(hand_results_det) < 1:
            hand_results_det = np.zeros([0,5])
        if len(item_results_det) < 1:
            item_results_det = np.zeros([0,5])
    else:
        hand_results_det = np.zeros([0,5])
        item_results_det = np.zeros([0,5])
    return human_results_det, hand_results_det, item_results_det

def obj_detector_image(detector, image, classes=None, conf_thresh=0.45, iou_thresh=0.3, device="gpu"):
        
    half = True
    if str(device) == "cpu":
        half = False

    img = image.copy()
    
    img = letterbox(img, new_shape=(640, 640))[0]
    
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Inference
    t1 = time_synchronized()
    with torch.no_grad():
        pred = detector(img, augment=False)[0]  # list: bz * [ (#obj, 6)]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes, agnostic=False)
    t2 = time_synchronized()
    fps = 1/(t2-t1)
    
    # Get final results
    results_det = pred[0]
    if results_det is not None and len(results_det):
        results_det[:, :4] = scale_coords(img.shape[2:], results_det[:, :4], image.shape).round()
        return results_det, fps
    else: 
        return torch.zeros((0, 5)), fps   