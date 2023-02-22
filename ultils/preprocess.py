

import sys
sys.path.insert(0, './detection/yolov7')

from typing import List, Dict
import logging
import torch
import numpy as np
import time
from ultils.object import Image

from detection.yolov7.utils.general import non_max_suppression, scale_coords
from detection.yolov7.utils.datasets import letterbox

from ultils.object import Human, Hand, Item, Box, Point, Image
from ultils.logger import set_logger
from config.config_common import CLASSES, objects
from config.config_detection import YoloConfig


# level = logging.NOTSET
# logger = set_logger(level=level)

class Detector:
    '''
    Detector:
    Args:
        weight_path: path of parameter file .pt or .pth
        model_load: Architecture of model    
    '''
    def __init__(self, model, device) -> None:
        self.device = device
        self.model = model.to(device)

    def preprocess_input(self, inputs):
        '''
        Args:
            input:
        Output:
            img process
        '''
        return inputs
    
    def preprocess_output(self, det, img_original:Image) -> List:
        '''
        pre-process rule, convert output to standart output
        Args:
            det: Object detected from detect function
        
        Output: List of object converted to standard.
        '''
        return det
    
    def detect(self, img):
        img_procced = self.preprocess_input(img)

        start_time = time.time()
        with torch.no_grad():
            det = self.model(img_procced)
        end_time = time.time()
        fps = round(1/(end_time-start_time), 3)

        det = self.preprocess_output(det=det, original_img=img)
        return det, fps


class Yolov7Detector(Detector):
    '''
    Yolov7 Detection model
    '''
    def __init__(self, model:str, device:str) -> None:
        super().__init__(model, device)
        # logger.info("Loading detector: Yolov7 ...")
    
    def preprocess_input(self, original_img:np.array) -> np.array:
        # half = True
        img = original_img.copy()
        img = letterbox(img, new_shape=(640, 640))[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img
    
    def preprocess_output(self, det, original_img) -> List:
        img = self.preprocess_input(original_img)
        pred = det[0]
        pred = non_max_suppression(pred, YoloConfig.conf_thres, YoloConfig.iou_thres, YoloConfig.classes, YoloConfig.agnostic)
        results_det = pred[0]
        if results_det is not None and len(results_det):
            results_det[:, :4] = scale_coords(img.shape[2:], results_det[:, :4], original_img.shape).round()
        
        # convert to object classification Human, Hand, remaining Items
        result_final = []
        for det in results_det.tolist():
            if det[5] == CLASSES.person:
                tl = Point(x=det[0], y=det[1])
                br = Point(x=det[2], y=det[3])
                result_final.append(
                    Human(
                        id=None,
                        id_object=CLASSES.person,
                        name_object=objects[CLASSES.person],
                        box=Box(tl=tl, br=br),
                        conf=det[4],
                        hands=[]
                        )
                    )
            elif det[5] == CLASSES.hand:
                tl = Point(x=det[0], y=det[1])
                br = Point(x=det[2], y=det[3])
                result_final.append(
                    Hand(
                        id=None,
                        id_object=CLASSES.hand,
                        name_object=objects[CLASSES.hand],
                        box=Box(tl=tl, br=br),
                        conf=det[4],
                        id_person=None
                    )
                )
            else:
                tl = Point(x=det[0], y=det[1])
                br = Point(x=det[2], y=det[3])
                result_final.append(
                    Item(
                        id=None, 
                        id_object=int(det[5]),
                        name_object=objects[int(det[5])],
                        box=Box(tl=tl, br=br),
                        conf=det[4]
                    )
                )
        # Clear result unnecessary
        del results_det
        pred.clear()
        return result_final

def load_model(weight_path, device):
    model = torch.load(weight_path, map_location=device)['model'].float().fuse().eval()
    if torch.cuda.is_available():
        model.to(device)
    return model

if __name__ == "__main__":

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_detect = load_model(weight_path=YoloConfig.weight_path, device=device)
    model_detect = Yolov7Detector(model=model_detect, device=device)
    
    # image = cv2.imread('/home/ubuntu/Pictures/download.jpeg')
    image = Image(id=0, img='/home/ubuntu/Pictures/download.jpeg', ratio=1)
    image_core = image.img
    result_dets = model_detect.detect(image_core)
    print(result_dets)

