from typing import List, Dict
from torch import Tensor
import cv2

class CLASSES:
    PERSON = {"id": 0, "label": "person"}
    OREO = {"id": 1, "label": "oreo_cake"}
    COFFEE = {"id": 2, "label": "coffee"}
    CHILLI_SAUCE = {"id": 3, "label": "chinsu"}
    COCA = {"id": 4, "label": "coca"}
    FANTA = {"id": 5, "label": "fanta"}
    HAND = {"id": 6, "label": "hand"}
box_img_area.shape[0]
class Color:
    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)

class Point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

class Box:
    def __init__(self, top_left:Point, bot_right:Point) -> None:
        self.top_left = top_left
        self.bot_right = bot_right
    
    def has_center(self) -> Point:
        x_cen = self.top_left.x + (self.bot_right.x-self.top_left.x)/2
        y_cen = self.top_left.y + (self.bot_right.y-self.top_left.y)/2
        return x_cen, y_cen
    
    def write_text(self,text:str, image:Tensor, size:float, color:dict):
        w = self.bot_right.x - self.top_left.x
        h = self.bot_right.y - self.top_left.y
        img_bbox = cv2.rectangle(image, (self.top_left.x, self.top_left.y), 
                                (self.bot_right.x, self.bot_right.y), 
                                (36,255,12), 
                                1)
        cv2.putText(img_bbox, text, (self.top_left.x, self.top_left.y-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        size, 
                                        color=color)
    


class Object:
    
    def __init__(self, cls_id, conf, box:Box, id=None) -> None:
        self.id = id # id tracking for human
        self.cls_id = cls_id
        self.box = box # top left bottom right
        self.conf = conf
        '''
            id: id_tracking
            cls_id: class_id fit CLASS object above
            conf: confidence object detection
            box: Box
        '''     
    
    def overlap_with(self, object, thres=0.5):
        # Compute S area 2 boxes
        S_self = (self.box.bot_right.x - self.box.top_left.x) * (self.box.bot_right.y - self.box.top_left.y)
        S_object = (object.box.bot_right.x - object.box.top_left.x)*(object.box.bot_right.y - object.box.top_left.y)

        # Compute coor overlap area
        xx = max(self.box.top_left.x, object.box.top_left.x)
        yy = max(self.box.top_left.y, object.box.top_left.y)
        aa = min(self.box.bot_right.x, object.box.bot_right.x)
        bb = min(self.box.bot_right.y, object.box.bot_right.y)

        # Compute S overlap area
        w = max(0, aa-xx)
        h = max(0, bb-yy)
        intersection_area = w*h

        # Compute S 2 boxes merge
        union_area = S_self + S_object - intersection_area
        Iou = intersection_area/union_area
        return Iou

class Item(Object):
    def __init__(self, cls_id, box: Box, conf) -> None:
        super().__init__(cls_id, box, conf)

class Hand(Object):
    def __init__(self, cls_id, box: Box, conf, id_person) -> None:
        super().__init__(cls_id, box, conf)
        self.id_person = id_person

    def touch(self, item:Item, thres=0.5):
        iou_score = self.overlap_with(item)
        return True if iou_score >= thres else False

class Human(Object):
    def __init__(self, cls_id, conf, box: Box, id, hands:List[Hand]) -> None:
        super().__init__(cls_id, conf, box, id)
        self.id = id
        self.id_object = CLASSES.PERSON
        self.hands = hands

