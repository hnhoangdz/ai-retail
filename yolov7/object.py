from typing import List, Dict

class CLASSES:
    PERSON = {"id": 0, "label": "person"}
    OREO = {"id": 1, "label": "oreo_cake"}
    COFFEE = {"id": 2, "label": "coffee"}
    CHILLI_SAUCE = {"id": 3, "label": "chinsu"}
    COCA = {"id": 4, "label": "coca"}
    FANTA = {"id": 5, "label": "fanta"}
    HAND = {"id": 6, "label": "hand"}

class Point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

class Box:
    def __init__(self, top_left:Point, bot_right:Point) -> None:
        self.top_left = top_left
        self.bot_right = bot_right
    
    def center_point(self):
        return (self.top_left.x + self.bot_right.x)/2, (self.top_left.y + self.bot_right.y)/2

class Object:
    
    def __init__(self, cls_id, box:Box, conf, id=None, frame_id=0)-> None:
        self.cls_id = cls_id
        self.box = box # top left bottom right
        self.conf = conf
        self.frame_id = frame_id
        '''
            id: id_tracking
            cls_id: class_id fit CLASS object above
            conf: confidence object detection
            box: Box
        '''     
    
    def overlap_with(self, object) -> float:
        
        # Compute S area 2 boxes
        # import ipdb; ipdb.set_trace()
        S_self = (self.box.bot_right.x - self.box.top_left.x)*(self.box.bot_right.y - self.box.top_left.y)
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
    def __init__(self, cls_id, box: Box, conf, frame_id) -> None:
        super().__init__(cls_id, box, conf, frame_id=frame_id)

class Hand(Object):
    def __init__(self, cls_id, box: Box, conf, id_person, frame_id) -> None:
        super().__init__(cls_id, box, conf, frame_id=frame_id)
        self.id_person = id_person

    def touch(self, item:Item, thresh=0.1):
        iou_score = self.overlap_with(item)
        return True if iou_score >= thresh else False

class Human(Object):
    def __init__(self, cls_id, box: Box, conf, id, hands:List[Hand], frame_id) -> None:
        super().__init__(cls_id, box, conf, id, frame_id)
        self.id = id
        self.hands = hands

