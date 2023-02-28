from helpers import draw_box, draw_line, load_model, bbox_to_center, search_area
import cfg
def which_area(human_obj):
    
    human_box = human_obj.box
    x1 = human_box.top_left.x
    y1 = human_box.top_left.y
    x2 = human_box.bot_right.x
    y2 = human_box.bot_right.y
    mid_leg_x = x1
    mid_leg_y = y2
    human_area = search_area(1920, 1080, mid_leg_x, mid_leg_y)
    
    if human_area == "shelf": 
        return human_area
    
    human_hands = human_obj.hands
    for hand in human_hands:
        hand_box = hand.box
        midx, midy = hand_box.center_point()
        hand_area = search_area(1920, 1080, midx, midy)
        if hand_area == "payment" and human_area == "payment":
            return human_area
    
    return "attend"

class Behavior:    
    # Identify behavior of people during consecutive 10 frames
    def __init__(self, consecutive_humans, consecutive_items) -> None:
        self.consecutive_humans = consecutive_humans # i -> i+10
        self.consecutive_items = consecutive_items # i -> i+10
        
    def get_item(self):
        pass
    
    def put_item_to_shelf(self):
        pass

    def bring_item_to_pay(self, current_state, items_on_shelf):
        pass

    def stolen(self):
        '''
           :))
        '''
        pass

class MmposeTrackBehavior(Behavior):
    def __init__(self, consecutive_humans, consecutive_items, last_human_ids) -> None:
        super().__init__(consecutive_humans, consecutive_items, last_human_ids)


class ByteTrackBehavior(Behavior):
    def __init__(self, consecutive_humans, consecutive_items) -> None:
        super().__init__(consecutive_humans, consecutive_items)
        self.consecutive_humans = consecutive_humans # i -> i+10
        self.consecutive_items = consecutive_items # i -> i+10

    def get_item(self):
        
        status = {}     
        
        # Iterate all humans id in current frame 
        # Check last frame
        for human_obj in self.consecutive_humans[-1]: 
            status[human_obj] = {
                "area": "",
                "items": [],
                "is_payed": False
            }
            # Identify currently where human is
            human_box = human_obj.box
            midx, midy = human_box.center_bot()
            area = which_area(human_obj)
            status[human_obj]["area"] = area
            in_area = True if area != "outside" else False
            
            if not in_area:
                continue
            
            # Identify currently what human touches
            human_hands = human_obj.hands
            duplicated_items = [False]*len(self.consecutive_items[-1])
            for hand_obj in human_hands:
                max_iou_hand = 0.0
                curr_item_obj = None
                item_idx = -1
                for i, item_obj in enumerate(self.consecutive_items[-1]):
                    if hand_obj.touch(item_obj) and hand_obj.iou_score >= max_iou_hand: 
                        max_iou_hand = hand_obj.iou_score
                        curr_item_obj = item_obj
                        item_idx = i
                if (item_idx != -1) and (duplicated_items[item_idx] == False):
                    status[human_obj]["items"].append(curr_item_obj)
                    duplicated_items[item_idx] = True

        # print("last status ", status)                                 
        # Check if last_human_obj got item in 10 consecutive frames or not 
        for last_human_obj in status:   
            in_area = True
            is_touching = False
            # Itearte each frame in the rest
            for frame_humans, frame_items in zip(self.consecutive_humans[:-1], 
                                                 self.consecutive_items[:-1]):              
                # Iterate humans object in each frame
                for human_obj in frame_humans:                 
                    if human_obj.id == last_human_obj.id:    
                        human_box = human_obj.box
                        midx, midy = human_box.center_bot()
                        area = which_area(human_obj)
                        in_area = True if area != "outside" else False
                        if not in_area:
                            break 
                        human_hands = human_obj.hands
                        for hand_obj in human_hands:
                            curr_item_obj = None
                            max_iou_hand = 0.0
                            for item_obj in frame_items:
                                if hand_obj.touch(item_obj) and hand_obj.iou_score >= max_iou_hand:
                                    curr_item_obj = item_obj
                                    max_iou_hand = hand_obj.iou_score
                            if curr_item_obj is not None:
                                for last_item_obj in status[last_human_obj]["items"]:
                                    if curr_item_obj.touch(last_item_obj, 0.5) and (curr_item_obj.cls_id == last_item_obj.cls_id):
                                        is_touching = True
                                        break
                            else:
                                is_touching = False
                            if is_touching:
                                break
                            
            if not (in_area and is_touching):
                status[last_human_obj]["items"] = []
        return status
    
    def put_item_to_shelf(self):
        pass

    def bring_item_to_pay(self, current_state, attend_storage, payment_storage):
        print("current_state ", current_state)
        for human_obj in current_state:
            human_area = current_state[human_obj]["area"]

            if human_area == "payment":
                current_items = current_state[human_obj]["items"]   

                for current_item_obj in current_items:
                    current_item_box = current_item_obj.box
                    midx, midy = current_item_box.center_point()
                    current_item_area = search_area(1920, 1080, midx, midy)
                
                    if current_item_area == "payment":

                        current_state[human_obj]["is_payed"] = True
                        human_id = human_obj.id

                        if human_id in attend_storage:
                            # Update attend storage
                            for i, attend_item_obj in enumerate(attend_storage[human_id]):
                                print("classes ", cfg.classes[attend_item_obj.cls_id])
                                if attend_item_obj.cls_id == current_item_obj.cls_id:
                                    if human_id not in payment_storage:
                                        payment_storage[human_id] = [attend_item_obj]
                                    else:
                                        payment_storage[human_id].append(attend_item_obj)
                                    attend_storage[human_id].pop(i) 

        return current_state, attend_storage, payment_storage

    def moving_attend_area(self, attend_storage):
        pass

    def stolen(self):
        '''
           :))
        '''
        pass