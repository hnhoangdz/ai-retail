from ultils.object import Human, Frame, Item, Polygon, KeyPoint, Point, Box
from typing import List, Dict
from config.config_common import AREA, NUM_FRAME_SET, Visualization
from ultils.common import convert2Real
import numpy as np
class Behavior:
    # Identify behavior of people during consecutive 10 frames
    def __init__(self, ) -> None:
        self.status = None
        pass
    
        
    def get_item(self):
        pass
    def put_item_to_shelf(self):
        pass

    def to_pay(self):
        pass
            

    def stolen(self):
        '''
           :))
        '''
        pass

class PoseBehavior(Behavior):
    def __init__(self) -> None:
        super().__init__()
        self.result = []

    def get_item(self, rs_num_frame_consecutive:List):
        '''
        validate item got?
        '''
        print(f"LEN HUMAN: {len(self.result)}")
        for rs_frame in rs_num_frame_consecutive:
            selection_area = [convert2Real(each[0], each[1], rs_frame.width, rs_frame.height) for each in AREA.selection]
            selection_area = [Point(x=each[0], y=each[1]) for each in selection_area]
            selection_area = Polygon(selection_area)
            line_shelve = [convert2Real(each[0], each[1], rs_frame.width, rs_frame.height) for each in AREA.line_shelve] 

            for human in rs_frame.humans:
                all_keypoint_hand = [kp for kp in human.left_hand_kp] + [kp for kp in human.right_hand_kp]
                # Validate human in selection area
                if human.stand_in(selection_area):
                    human.flag_in_selection_area = True
                
                # Validate hand over line shelve
                for kp in all_keypoint_hand:
                    if kp.over_line_has_2_point(point1=line_shelve[0], point2=line_shelve[1]):
                        human.flag_hand_over_line = True
                        break
                for item in rs_frame.items:
                    cnt_kp = 0
                    for kp in all_keypoint_hand:
                        if kp.in_box(item.box):
                            cnt_kp += 1
                        if cnt_kp > 1/9*len(all_keypoint_hand):
                            human.cnt_hand_touch_item += 1
                            break
                    # check condition to add item to human
                    # if human.flag_in_selection_area and human.flag_hand_over_line and human.cnt_hand_touch_item > 1/6*NUM_FRAME_SET:
                    if human.flag_in_selection_area and human.cnt_hand_touch_item > 1/6*NUM_FRAME_SET:
                        # print("DA LAY VAT")
                        if item not in human.item_holding:
                            human.item_holding.append(item)
                            human.status = 'holding_item'
                if not human in self.result:
                    self.result.append(human)
                
                if len(self.result) != 0:
                    for rs_human in self.result:
                        if rs_human == human and len(human.item_holding) > len(rs_human.item_holding):
                            for rs_item in human.item_holding:
                                if not rs_item in rs_human.item_holding:
                                    rs_human.item_holding.append(rs_item)
            
        # for human in self.result:
        #     print(f"frame id: {rs_frame.id} -------human_id: {human.track_id} -------holding: {human.item_holding}")

        # print("======================================================================")
        
                            
    def to_pay(self, rs_num_frame_consecutive, all_human_in_retail):
        for rs_frame in rs_num_frame_consecutive:
            payment_area = [convert2Real(each[0], each[1], rs_frame.width, rs_frame.height) for each in AREA.payment]
            payment_area = [Point(x=each[0], y=each[1]) for each in payment_area]
            payment_area = Polygon(payment_area)
            for rs_human in rs_frame.humans:
                rs_human.cnt_in_area_pay+=1 if rs_human.stand_in(payment_area) else 0
                if rs_human.cnt_in_area_pay > 3:
                    for human in all_human_in_retail:
                        if human == rs_human and len(human.item_holding) > 0:
                            human.was_paid = True
                            break
        for human in all_human_in_retail:
            print(f"-------human_id: {human.track_id} -------was paid: {human.was_paid}   -------holding: {human.item_holding}")

        print("======================================================================")
        

        


