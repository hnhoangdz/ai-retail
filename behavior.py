from helpers import draw_box, draw_line, load_model, bbox_to_center, search_area



class Behavior:    # Identify behavior of people during consecutive 10 frames
    def __init__(self, consecutive_humans, consecutive_items, last_human_ids) -> None:
        self.consecutive_humans = consecutive_humans # i -> i+10
        self.consecutive_items = consecutive_items # i -> i+10
        self.last_human_ids = last_human_ids
        
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
    def __init__(self, consecutive_humans, consecutive_items, last_human_ids) -> None:
        super().__init__(consecutive_humans, consecutive_items, last_human_ids)
        self.consecutive_humans = consecutive_humans # i -> i+10
        self.consecutive_items = consecutive_items # i -> i+10
        self.last_human_ids = last_human_ids
        
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
            midx, midy = human_box.center_point()
            area = search_area(1920, 1080, midx, midy)
            status[human_obj]["area"] = area
            in_area = True if area != "outside" else False
            if not in_area:
                continue
            # Identify currently what human touches
            human_hands = human_obj.hands
            for hand_obj in human_hands:
                for item_obj in self.consecutive_items[-1]:
                    if hand_obj.touch(item_obj):
                        status[human_obj]["items"].append(item_obj)
                                                
        print("last status ", status)   
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
                        print("vclllllllllllll ")
                        human_box = human_obj.box
                        midx, midy = human_box.center_point()
                        area = search_area(1920, 1080, midx, midy)
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

    def bring_item_to_pay(self, current_state, items_on_shelf):
        for human in current_state:
            on_store = False
            if current_state[human]["area"] == "payment":
                items_obj = current_state[human]["items"]
                for item in items_obj:
                    on_store = item.cls_id in items_on_shelf
            current_state[human]["is_payed"] = on_store
        return current_state
            

    def stolen(self):
        '''
           :))
        '''
        pass

