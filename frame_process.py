from helpers import draw_box, draw_line, load_model, bbox_to_center, search_area
from cfg import *
from yolov7.object import Item, Hand, Human, Object, Box, Point
from typing import Dict, List

from yolov7.detect_inf import obj_detector

import cv2
import numpy as np

class Frame(object):
    def __init__(self, ith, frame, detector, tracker, display_area=True):
        self.ith = ith
        self.frame = frame
        self.detector = detector
        self.tracker = tracker
        self.org_h, self.org_w = self.frame.shape[:2]
        self.frame_objs = list()
        
        if display_area:
            self.__draw_area()
        
    def __draw_area(self):
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
        self.classes_id = self.boxes[:, 5]
        
        # Get Item Objects
        for i, box in enumerate(self.boxes):
            if self.classes_id[i] != 0 and self.classes_id[i] != 6:
                i_box = Box(Point(box[0], box[1]), Point(box[2], box[3]))
                i_obj = Object(self.classes_id[i], box[4], i_box)
                self.frame_objs.append(i_obj)
        # self.human_boxes = self.boxes[self.classes_id == 0]
        # self.hand_boxes = self.boxes[self.classes_id == 6]
        # if len(self.human_boxes) < 1:
        #     self.hand_boxes = None
        # self.item_boxes = self.boxes[self.classes_id in np.arange(1, 6)]
        
    def tracking(self, input_size, aspect_ratio_thresh, min_box_area, visualize=True):
        
        online_targets = self.tracker.update(self.boxes[self.classes_id == 0], self.frame.shape[:2], input_size)
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
        
        if len(online_tlwhs) == 0:
            return None
        
        # Get Person Objects
        for i, p_box in enumerate(online_tlwhs):
            p_box = Box(Point(p_box[0], p_box[1]), Point(p_box[2], p_box[3]))
            p_obj = Object(0, online_scores[i], p_box, online_ids[i])
            self.frame_objs.append(p_obj)
        
        # Get Hand Objects
        for i, box in enumerate(self.boxes):
            if self.classes_id[i] == 6:
                if len(online_tlwhs) < 1:
                    continue
                else:
                    h_box = Box(Point(box[0], box[1]), Point(box[2], box[3]))
                    h_obj = Object(6, box[4], h_box)
                    self.frame_objs.append(h_obj)
                
        # for i, cls_id in enumerate(self.classes_id):
        #     if cls_id == 0:
        #         self.frame_objs.append(Object(cls_id, ))
        # return np.concatenate((np.array(online_tlwhs), np.array(online_ids)), axis=1) 
        
                
    
    
    
        
    
    
    
        
    