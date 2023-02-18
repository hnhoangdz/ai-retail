from typing import List, Dict
class CLASSES:
    PERSON = {"id": 0, "label": "person"}
    OREO = {"id": 1, "label": "oreo_cake"}
    COFFEE = {"id": 2, "label": "coffee"}
    CHINSU = {"id": 3, "label": "chinsu"}
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

class Object:
    def __init__(self, id, id_object, name_object, box:Box, conf) -> None:
        self.id = id
        self.id_object = id_object
        self.name_object = name_object
        self.box = box #top left bottom right
        self.conf = conf
    
    def overlap_with(self, object:Object, thres=0.5):
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
    def __init__(self, id, id_object, name_object, box: Box, conf) -> None:
        super().__init__(id, id_object, name_object, box, conf)


class Hand(Object):
    def __init__(self, id, id_object, name_object, box: Box, conf, id_person) -> None:
        super().__init__(id, id_object, name_object, box, conf)
        self.id_person = id_person

    def touch(self, item:Item, thres=0.5):
        iou_score = self.overlap_with(item)
        return True if iou_score >= thres else False


class Human(Object):
    def __init__(self, id, id_object, name_object, box: Box, conf, hands:List[Hand]) -> None:
        super().__init__(id, id_object, name_object, box, conf)
        self.id = id
        self.id_object = CLASSES.PERSON
        self.hands = hands

