from typing import List, Dict
from object import Human, Hand, Item, Object, Box
from torch import Tensor
from helpers import draw_box, draw_line, load_model, bbox_to_center, search_area
from yolov7.detect_inf import obj_detector
from cfg import *
import cv2
import numpy as np


class Frame:
    def __init__(self, image:Tensor, detector, tracker, humans:List[Human], hands:List[Hand], items:List[Item]) -> None:
        self.detector = detector
        self.tracker = tracker
        self.image = image
        self.humans = []
        self.items = []
        self.hands = []
        
    def predict(self) -> List[Object]:
        pass
    
    def cap_shelve_area(self) -> Box:
        shelve_box = Box()
        return shelve_box
    
    def cap_human_in_shelve_area(self) -> Box:
        human_in_shelve_are = Box()
        return human_in_shelve_are

    def cap_common_area(self) -> Box:
        common_area = Box()
        return common_area

    def cap_pay_area(self) -> Box:
        pay_area = Box
        return pay_area
    
    def draw_bbox_area(self):
        # Draw shelf-area
        draw_line(self.frame, 0.2, 0.43, 0.39, 0.2) # t
        draw_line(self.frame, 0.39, 0.2, 0.44, 0.4) # r
        draw_line(self.frame, 0.44, 0.4, 0.29, 0.64) # b
        draw_line(self.frame, 0.2, 0.43, 0.29, 0.64) # l
        
        # Draw attend-area
        draw_line(self.frame, 0.14, 0.5, 0.4, 0.17) # t
        draw_line(self.frame, 0.4, 0.17, 0.73, 0.33) # r 
        draw_line(self.frame, 0.58, 0.97, 0.73, 0.33) # b
        draw_line(self.frame, 0.14, 0.5, 0.58, 0.97) # l

    def which_human_area(self, box):
        midx, midy = bbox_to_center(box)
        area = search_area(self.frame, midx, midy)
        return area
    

    def detect(self, conf_thresh, iou_thresh, device, classes = None):
        
        self.boxes, _ = obj_detector(self.detector, 
                                  self.frame,
                                  classes,
                                  conf_thresh, 
                                  iou_thresh,
                                  device)
        
        self.boxes = self.boxes.detach().cpu().numpy()
        classes_id = self.boxes[:, 5]
        self.human_boxes = self.boxes[classes_id == 0]
        self.hand_boxes = self.boxes[classes_id == 6]
        if len(self.human_boxes) < 1:
            self.hand_boxes = None
        self.item_boxes = self.boxes[classes_id in np.arange(1, 6)]
        
        return self.human_boxes, self.hand_boxes, self.item_boxes
    
    def tracking(self, boxes, input_size, aspect_ratio_thresh, min_box_area, visualize=True):
        
        online_targets = self.tracker.update(boxes, self.frame.shape[:2], input_size)
        input_h, input_w = input_size
        scale = min(input_h / float(self.org_h), input_w / float(self.org_w))
        
        online_tlwhs = []
        online_ids = []
        online_scores = []
        
        # 
        for track in online_targets:
            tlwh = track.tlwh
            tid = track.track_id
            vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
            if (tlwh[2] * tlwh[3] > min_box_area) and not vertical:
                tlwh = tlwh * scale
                online_tlwhs.append([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]])
                online_ids.append([tid])
                online_scores.append(track.score)
        
        if visualize:
            for idx, box in enumerate(online_tlwhs):
                x1, y1, x2, y2 = box
                cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(self.frame, str(online_ids[idx]), (int(x1), int(y1-7)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        print(np.array(online_tlwhs).shape)
        print(np.array(online_ids).T.shape)
        return np.concatenate((np.array(online_tlwhs), np.array(online_ids).T), axis=1) 
        