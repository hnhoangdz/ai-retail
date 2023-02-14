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

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative     
                                                                               
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

def obj_detector(detector, image, classes=None, conf_thresh=0.45, iou_thresh=0.3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    detector.to(device)
    
    half = False
    if str(device) != "cpu":
        half = True
    img = image.copy()
    
    img = letterbox(img, new_shape=(1024, 1024))[0]
    
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

if __name__ == "__main__":
    
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    
    model = load_detect_model("/home/hoangdinhhuy/hoangdinhhuy/VTI/retail_store/yolov7/trained_models/yolov7.pt")
    video = cv2.VideoCapture("/home/hoangdinhhuy/hoangdinhhuy/VTI/retail_store/video_duy.mp4")
    while True:
        t0 = time.time()
        ret, frame = video.read()

        result, fps = obj_detector(model, frame)
        
        MARGIN=10
        for (xmin, ymin, xmax,   ymax,  confidence,  clas) in result:
            with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
                results = pose.process(frame[int(ymin)+MARGIN:int(ymax)+MARGIN,int(xmin)+MARGIN:int(xmax)+MARGIN:])
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame[int(ymin)+MARGIN:int(ymax)+MARGIN,int(xmin)+MARGIN:int(xmax)+MARGIN:], 
                                              results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    for id, lm in enumerate(results.pose_landmarks.landmark):
                        h, w,c = frame[int(ymin)+MARGIN:int(ymax)+MARGIN,int(xmin)+MARGIN:int(xmax)+MARGIN:].shape
                        print(id, lm)
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        cv2.circle(frame[int(ymin)+MARGIN:int(ymax)+MARGIN,int(xmin)+MARGIN:int(xmax)+MARGIN:], (cx, cy), 5, (255,0,0), cv2.FILLED)

        print(1/(time.time() - t0))
        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    video.release()
    cv2.destroyAllWindows()