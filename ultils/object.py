from typing import List, Dict
from config.config_common import COLOR, object, CLASSES
import cv2
class Point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
    
    def inside(self, polygon:Polygon) -> bool:
        '''Validate point inside a polygon area'''
        return cv2.pointPolygonTest(contour=polygon.points, pt=(self.x, self.y), measureDist=True)


class Box:
    def __init__(self, tl:Point, br:Point) -> None:
        self.tl = tl
        self.br = br

    def has_center(self) -> Point:
        '''return a center point a bounding box'''
        return Point((self.tl.x+self.br.x)//2, (self.tl.y+self.br.y)//2)
    
    def draw_box_in(self, image, color:tuple=COLOR.green, thickness:int=1, label:str='') -> None:
        '''
        draw object bounding box.
        '''
        const_edge_len = 50
        # top left
        cv2.line(image, 
                (self.box.tl.x, self.box.tl.y), 
                (self.box.tl.x, self.box.tl.y + const_edge_len), 
                color, thickness) 
        cv2.line(image, 
                (self.box.tl.x, self.box.tl.y),
                (self.box.tl.x + const_edge_len, self.box.tl.y),
                color, thickness) 

        # bottom_right
        cv2.line(image, 
                (self.box.br.x, self.box.br.y),
                (self.box.br.x - const_edge_len, self.box.br.y),
                color, thickness) 
        cv2.line(image, 
                (self.box.br.x, self.box.br.y),
                (self.box.br.x, self.box.br.y - const_edge_len),
                color, thickness) 

        # put label text of the bounding box
        cv2.putText(image, label,
                    (self.box.tl.x, self.box.tl.y-15),
                    fontScale = 0.8,
                    color=color,
                    thickness=thickness,
                    fontFace=cv2.LINE_AA
                    )


class Polygon():
    def __init__(self, points:List[Point]) -> None:
        self.points = [(pt.x, pt.y) for pt in self.points]
    
    def draw_polygon_in(self, image, color:tuple=COLOR.yellow, thickness:int=1, label:str='') -> None:
        '''draw polygon in frame'''
        # drawing...
        cv2.polylines(image=image, pts=self.points, color=color, thickness=thickness, isClosed=True)
        
        # draw_label
        top_point = min(self.points, key=lambda x:x[1])
        cv2.putText(image, label,
                    (top_point[0], top_point[1]-15),
                    fontScale = 0.8,
                    color=color,
                    thickness=thickness,
                    fontFace=cv2.LINE_AA
                    )


class Object:
    def __init__(self, id, id_object, name_object, box:Box, conf) -> None:
        self.id = id
        self.id_object = id_object
        self.name_object = name_object
        self.box = box # top left bottom right
        self.conf = conf # confidence of object
        self.local = None
    
    def overlap_with(self, object:Object, thres=0.5) -> float:
        # Compute S area 2 boxes
        S_self = (self.box.br.x - self.box.tl.x) * (self.box.br.y - self.box.tl.y)
        S_object = (object.box.br.x - object.box.tl.x)*(object.box.br.y - object.box.tl.y)

        # Compute coor overlap area
        xx = max(self.box.tl.x, object.box.tl.x)
        yy = max(self.box.tl.y, object.box.tl.y)
        aa = min(self.box.br.x, object.box.br.x)
        bb = min(self.box.br.y, object.box.br.y)

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

    def touch(self, item:Item, thres=0.5) -> bool:
        iou_score = self.overlap_with(item)
        return True if iou_score >= thres else False


class Human(Object):
    def __init__(self, id, id_object, name_object, box: Box, conf, hands:List[Hand]) -> None:
        super().__init__(id, id_object, name_object, box, conf)
        self.id = id
        self.id_object = CLASSES.PERSON
        self.hands = hands


# if __name__ == "__main__":
    # pass
    