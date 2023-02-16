from helpers import (draw_box, 
                     draw_line, 
                     load_model, 
                     bbox_to_center, 
                     search_area,
                     get_max_iou,
                     get_iou)
from cfg import *
from yolov7.object import Item, Hand, Human, Object, Box, Point

from yolov7.detect_inf import obj_detector

import cv2
import numpy as np

class Frame(object):
    def __init__(self, ith, frame, detector, tracker, display_area=True):
        self.ith = ith
        self.frame = frame
        self.frame_copy = self.frame.copy()
        self.detector = detector
        self.tracker = tracker
        self.org_h, self.org_w = self.frame.shape[:2]
        
        self.frame_objs = {"humans": [], 
                           "items": []
                           }
        
        self.is_detected = True
        
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
        area = search_area(1920, 1080, midx, midy)
        return area
    
    def get_items_on_shelf(self):
        count = 0
        for box in self.items_boxes:
            if self.which_human_area(box[:4]) == "shelf":
                count += 1
        if count == 5:
            return self.items_boxes[:, 5]
        return None
    
    def detect(self, conf_thresh, iou_thresh, device, classes = None, visualize=True):
        
        self.boxes, _ = obj_detector(self.detector, 
                                  self.frame_copy,
                                  classes,
                                  conf_thresh, 
                                  iou_thresh,
                                  device)
        
        self.boxes = self.boxes.detach().cpu().numpy()    
        
        if len(self.boxes) == 0:
            self.is_detected = False
            
        else:
            
            self.classes_id = self.boxes[:, 5]
            
            self.human_boxes = self.boxes[self.classes_id == 0]
            self.hands_boxes = self.boxes[self.classes_id == 6]
            self.items_boxes = self.boxes[(self.classes_id != 0) & (self.classes_id != 6)]

            
            if visualize:
                for idx, box in enumerate(self.human_boxes):
                    cv2.rectangle(self.frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2) 
            
            # Get Item Objects
            for i, box in enumerate(self.boxes):
                if self.classes_id[i] != 0 and self.classes_id[i] != 6:
                    i_box = Box(Point(box[0], box[1]), Point(box[2], box[3]))
                    i_obj = Item(self.classes_id[i], i_box, box[4], frame_id=self.ith)
                    self.frame_objs["items"].append(i_obj)
                
    def tracking(self, input_size, aspect_ratio_thresh, min_box_area, visualize=True):
        
        if len(self.human_boxes) > 1:
            
            # Ensure humans stand in proposal areas
            box_area = []
            for box in self.human_boxes:
                area = self.which_human_area(box[:4])
                box_area.append(area)
            box_area = np.array(box_area)
            self.human_boxes = self.human_boxes[box_area != "outside"]
            
            # tracking
            online_targets = self.tracker.update(self.human_boxes, self.frame.shape[:2], input_size)
            input_h, input_w = input_size
            scale = min(input_h / float(self.org_h), input_w / float(self.org_w))
            
            online_tlwhs = []
            online_ids = []
            online_scores = []
            
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
                    x1, y1, x2, y2 = box[:4]
                    cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(self.frame, str(online_ids[idx]), (int(x1), int(y1-7)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                
                for idx, box in enumerate(self.hands_boxes):
                    cv2.rectangle(self.frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                    
            # after tracking, confirm there are humans available in proposal area
            if len(online_tlwhs) > 0:
                
                self.human_boxes = np.concatenate((np.array(online_tlwhs), np.array(online_ids)), axis=1) 
                
                # find hands belong to who
                if len(self.hands_boxes) > 0:
                    self.whose_hand()
                
                # append human objects 
                for i, box in enumerate(self.human_boxes):
                    
                    p_box = Box(Point(box[0], box[1]), Point(box[2], box[3]))
                    p_id = box[-1]
                    hands = self.hands_boxes[self.hands_boxes[:,-1] == p_id]
                    hands_list = list()
                    
                    for h in hands:
                        h_box = Box(Point(h[0], h[1]), Point(h[2], h[3]))
                        h_obj = Hand(6, h_box, h[4], h[-1], frame_id=self.ith)
                        hands_list.append(h_obj)
                    p_obj = Human(0, p_box, online_scores[i], online_ids[i][0], hands_list, frame_id=self.ith)
                    
                    self.frame_objs["humans"].append(p_obj)
    
    def whose_hand(self):
        
        self.hands_boxes = np.concatenate((self.hands_boxes, np.array([[-1]*len(self.hands_boxes)]).T), axis=1)

        count = {}

        for human_box in self.human_boxes:
            count[human_box[-1]] = 0
                
        for i, hand_box in enumerate(self.hands_boxes):
            
            tmp_id = -1
            tmp_overlap = 0.0
            
            for human_box in self.human_boxes:
                
                if count[human_box[-1]] == 2:
                    continue
                
                overlap_area = get_iou(hand_box[:4], human_box[:4])
                if overlap_area > tmp_overlap:
                    tmp_overlap = overlap_area
                    tmp_id = human_box[-1]
                    
            self.hands_boxes[i][-1] = tmp_id
            if tmp_id != -1:
                count[tmp_id] += 1
        
                
    
    
    
        
    
    
    
        
    