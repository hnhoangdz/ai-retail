from helpers import (draw_box, 
                     draw_line, 
                     load_model, 
                     bbox_to_center, 
                     search_area,
                     get_max_iou,
                     get_iou)
import cfg
from cfg import *
from yolov7.object import Item, Hand, Human, Object, Box, Point

from yolov7.detect_inf import obj_detector
from helpers import draw_box, draw_line, load_model
import time
import cv2
import numpy as np

class Frame(object):
    def __init__(self, ith, frame, human_detector, hands_items_detector, tracker, display_area=False):
        self.ith = ith
        self.frame = frame
        self.frame_copy = self.frame.copy()
        self.human_detector = human_detector
        self.hands_items_detector = hands_items_detector
        self.tracker = tracker
        self.org_h, self.org_w = self.frame.shape[:2]
        
        self.frame_objs = {"humans": [], 
                           "items": []
                           }
        
        self.is_detected = True
        
        if display_area:
            self.__draw_area()
        
    def __draw_area(self):
        
        # In-out
        draw_line(self.frame, in_out_area[0][0], in_out_area[0][1], 
                              in_out_area[1][0], in_out_area[1][1])
        
        # Shelf
        draw_line(self.frame, shelf_area[0][0], shelf_area[0][1], 
                              shelf_area[1][0], shelf_area[1][1])
        # Payment
        draw_line(self.frame, payment_area[0][0], payment_area[0][1], 
                              payment_area[1][0], payment_area[1][1])
        
        draw_line(self.frame, payment_area[1][0], payment_area[1][1], 
                              payment_area[2][0], payment_area[2][1])
        
        draw_line(self.frame, payment_area[2][0], payment_area[2][1], 
                              payment_area[3][0], payment_area[3][1])
        
        draw_line(self.frame, payment_area[0][0], payment_area[0][1], 
                              payment_area[3][0], payment_area[3][1])
    
    def box_area(self, box):
        x1, y1, x2, y2 = box[:4]
        mid_leg_x = (x1+x2)/2
        mid_leg_y = y2
        area = search_area(1920, 1080, mid_leg_x, mid_leg_y)
        return area
    
    def which_area(self, human_obj):

        human_box = human_obj.box
        x1 = human_box.top_left.x
        y1 = human_box.top_left.y
        x2 = human_box.bot_right.x
        y2 = human_box.bot_right.y
        mid_leg_x = (x1+x2)/2
        mid_leg_y = y2
        human_area = search_area(1920, 1080, mid_leg_x, mid_leg_y)
        
        if human_area == "shelf" or human_area == "payment":
            return human_area
        
        human_hands = human_obj.hands
        for hand in human_hands:
            hand_box = hand.box
            midx, midy = hand_box.center_point()
            hand_area = search_area(1920, 1080, midx, midy)
            if hand_area == "shelf" or hand_area == "payment":
                return hand_area
        
        return human_area
    
    def detect(self, conf_thresh, iou_thresh, device, classes = None, visualize=False):
        
        self.human_boxes, self.hands_boxes, self.items_boxes = obj_detector(self.human_detector, 
                                                                            self.hands_items_detector,
                                                                            self.frame_copy,
                                                                            classes,
                                                                            conf_thresh, 
                                                                            iou_thresh,
                                                                            device)  
        if not visualize:
            # for box in self.human_boxes:
            #     x1,y1,x2,y2 = box[:4]
            #     cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2, cv2.LINE_AA)
            # for box in self.hands_boxes:
            #     x1,y1,x2,y2 = box[:4]
            #     cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2, cv2.LINE_AA)
            for box in self.items_boxes:
                x1,y1,x2,y2 = box[:4]
                cls_id = int(box[-1])
                cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(self.frame, cfg.classes[cls_id], (int(x1), int(y1-20)), cv2.LINE_AA, 0.7, (0,255,0))
        if (len(self.human_boxes) < 1):
            self.is_detected = False

        if len(self.items_boxes) >= 1:
            # Get Item Objects
            for i, box in enumerate(self.items_boxes):
                i_box = Box(Point(box[0], box[1]), Point(box[2], box[3]))
                i_obj = Item(int(box[-1]), i_box, box[4], frame_id=self.ith)
                self.frame_objs["items"].append(i_obj)
                
    def tracking(self, input_size, aspect_ratio_thresh, min_box_area, visualize=False):
        
        if len(self.human_boxes) >= 1:
            
            # Ensure humans stand in proposal areas
            box_area = []
            for box in self.human_boxes:
                area = self.box_area(box[:4])
                print("area ", area)
                box_area.append(area)
            box_area = np.array(box_area)
            self.human_boxes = self.human_boxes[box_area != "outside"]
            
            # Tracking
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

                if len(self.hands_boxes) > 0:
                    for idx, box in enumerate(self.hands_boxes):
                        cv2.rectangle(self.frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

            # after tracking, confirm there are humans available in proposal area
            if len(online_tlwhs) > 0:
                
                self.human_boxes = np.concatenate((np.array(online_tlwhs), \
                                                   np.array(online_ids)), \
                                                   axis=1) 
                # find hands belong to who
                if len(self.hands_boxes) > 0:
                    self.whose_hand()
                
                # append human objects 
                for i, box in enumerate(self.human_boxes):
                    
                    p_box = Box(Point(box[0], box[1]), Point(box[2], box[3]))
                    p_id = box[-1]
                    hands_list = list()
                    if len(self.hands_boxes) > 0:
                        hands = self.hands_boxes[self.hands_boxes[:,-1] == p_id]
                        
                        for h in hands:
                            h_box = Box(Point(h[0], h[1]), Point(h[2], h[3]))
                            h_obj = Hand(2, h_box, h[4], h[-1], frame_id=self.ith)
                            hands_list.append(h_obj)
                        
                    p_obj = Human(0, p_box, online_scores[i], online_ids[i][0], hands_list, frame_id=self.ith)
                    
                    self.frame_objs["humans"].append(p_obj)
    
    def whose_hand(self):

        # Find hands belong to who
        self.hands_boxes = np.concatenate((self.hands_boxes, \
                                           np.array([[-1]*len(self.hands_boxes)]).T), \
                                           axis=1)
        
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
        
                
    
    
    
        
    
    
    
        
    