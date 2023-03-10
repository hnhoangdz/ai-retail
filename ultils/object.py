from typing import List, Dict, Tuple
from config.config_common import COLOR, objects, CLASSES
import cv2
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq
import math
import os
from keypoint import keypoint_definition

class Point:
    def __init__(self, x, y) -> None:
        self.x = int(x)
        self.y = int(y)
    
    def inside(self, polygon) -> bool:
        '''Validate point inside a polygon area'''
        # import ipdb; ipdb.set_trace()
        coutour = np.array(polygon.points).reshape(4, 1, 2)
        score = cv2.pointPolygonTest(contour=coutour, pt=(self.x, self.y), measureDist=True)
        return True if score > 0 else False
    
    def over_line_has_2_point(self, point1:Tuple, point2:Tuple):
        points = [point1, point2]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        x_at_line = (self.y - c)//m
        return True if self.x < x_at_line else False

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
        self.points = [(pt.x, pt.y) for pt in points]
    
    def draw(self, image:np.array, color:tuple=COLOR.yellow, thickness:int=1, label:str='') -> None:
        '''draw polygon in frame'''
        # drawing...
        points = np.array(self.points)
        cv2.polylines(img=image, pts=[points], isClosed=True, color=color, thickness=thickness)
        
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
    
    def overlap_with(self, object, thres=0.5) -> float:
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
    
    

    def visual(self, image:np.array, color=COLOR.blue, thickness=2, label:str=''):
        # print(f"img_type: {type(image)}     color: {color}     tl_x: {self.box.tl.x}     tl_y: {self.box.tl.y}     br_x: {self.box.br.x}      br_y: {self.box.br.y}")
        cv2.rectangle(image,
                    (self.box.tl.x, self.box.tl.y),
                    (self.box.br.x, self.box.br.y),
                    color=color,
                    thickness=thickness
                    )
        cv2.putText(image, label,
                (self.box.tl.x, self.box.tl.y-15),
                fontScale = 0.8,
                color=color,
                thickness=thickness,
                fontFace=cv2.LINE_AA
                )


class Item(Object):
    def __init__(self, id, id_object, name_object, box: Box, conf) -> None:
        super().__init__(id, id_object, name_object, box, conf)

    def __eq__(self, __o: object) -> bool:
        return self.id_object == __o.id_object


class Hand(Object):
    def __init__(self, id, id_object, name_object, box: Box, conf, id_person) -> None:
        super().__init__(id, id_object, name_object, box, conf)
        self.id_person = id_person

    def touch(self, item:Item, thres=0.5) -> bool:
        iou_score = self.overlap_with(item)
        return True if iou_score >= thres else False


class Human(Object):
    def __init__(self, track_id, id_object, name_object, box: Box, conf) -> None:
        super().__init__(id, id_object, name_object, box, conf)
        self.track_id = track_id
        self.id_object = None
        self.left_hand_kp = []
        self.right_hand_kp = []
        self.left_leg_kp = []
        self.right_leg_kp = []
        self.item_holding = []
        self.area = None
        self.hold_item_flag = True if len(self.item_holding)>0 else False 
        self.flag_in_selection_area = False
        self.flag_hand_over_line = False
        self.cnt_hand_touch_item = 0
        self.payment_flag = False
        self.cnt_in_area_pay = 0
        self.was_paid = False
    
    def __eq__(self, __o: object) -> bool:
        return self.track_id == __o.track_id
    
    def stand_in(self, area:Polygon) -> bool:
        '''
        human stand in area and not?
        '''
        cen_point_2_leg = Point(x=(self.left_leg_kp[0].x+self.right_leg_kp[0].x)//2, y=(self.left_leg_kp[0].y+self.right_leg_kp[0].y)//2)
        return cen_point_2_leg.inside(area) 

    
    def hold(self, item:Item, thres:float) -> List[Item]:
        item_held = []
        l_cnt = 0
        for kp_left_hand in self.left_hand_kp:
            if kp_left_hand.in_box(item.box):
                l_cnt += 1
        
        r_cnt = 0
        for kp_right_hand in self.right_hand_kp:
            if kp_right_hand.in_box(item.box):
                r_cnt += 1
        
        if l_cnt/len(self.left_hand_kp)>=thres or r_cnt/len(self.right_hand_kp)>=thres:
            for item_held in item_held:
                if item_held.box.tl.x != item.box.tl.x:
                    item_held.append(item)

        return item_held


    
    def visual(self, image: np.array, color=COLOR.blue, thickness=2, label: str = ''):
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
        # visual Label
        # import ipdb; ipdb.set_trace()
        cv2.putText(image, label,
                    (self.box.tl.x, self.box.tl.y-15),
                    fontScale = 0.8,
                    color=color,
                    thickness=thickness,
                    fontFace=cv2.LINE_AA
                    )
        


class Frame:
    def __init__(self, id, img, ratio) -> None:
        if isinstance(img, str):
            self.path = img
            self.name = os.path.basename(img)
            self.img = cv2.imread(img)
        else:
            self.img = img
            self.path = None
            self.name = None
        self.id = id
        self.shape = self.img.shape
        self.width = self.shape[1]
        self.height = self.shape[0]
        self.channel = self.shape[2] if len(self.shape) > 2 else None
        self.ratio = ratio
        self.humans = []
        self.items = []
    

    def has_area(self, ratio_points:Tuple) -> Polygon:
        real_point_obj = []
        for each in ratio_points:
            x_real = each[0] * self.width
            y_real = each[1] * self.height
            real_point_obj.append(Point(x=x_real, y=y_real))
        return Polygon(points=real_point_obj)


class KeyPoint(Point):
    def __init__(self, x, y, conf:float, id_:int, name:str, color:tuple, type_:str, swap:str) -> None:
        super().__init__(x, y)
        self.conf = conf
        self._id = id_
        self.name = name
        self.color = color
        self.type = type_
        self.swap = swap

    def in_box(self, box:Box) -> bool:
        return self.x > box.tl.x and self.y > box.tl.y and self.x < box.br.x and box.br.y


        
# if __name__ == "__main__":
    # pass
    