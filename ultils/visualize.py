from yolov7.object import Human, Item, CLASSES
from helpers import draw_line
import cv2
from typing import Dict, List
from cfg import COLOR, classes
import cfg

def visualize_human(human:Human, image, color:tuple, thickness, label, area):
    # visual Box
    const_edge_len = 50
    # top left
    cv2.line(image, 
            (human.box.top_left.x, human.box.top_left.y), 
            (human.box.top_left.x, human.box.top_left.y + const_edge_len), 
            color, thickness) 

    cv2.line(image, 
            (human.box.top_left.x, human.box.top_left.y),
            (human.box.top_left.x + const_edge_len, human.box.top_left.y),
            color, thickness) 

    # bottom_right
    cv2.line(image, 
            (human.box.bot_right.x, human.box.bot_right.y),
            (human.box.bot_right.x - const_edge_len, human.box.bot_right.y),
            color, thickness) 
    
    cv2.line(image, 
            (human.box.bot_right.x, human.box.bot_right.y),
            (human.box.bot_right.x, human.box.bot_right.y - const_edge_len),
            color, thickness) 
    # visual Label
    # import ipdb; ipdb.set_trace()
    cv2.putText(image, label,
                (human.box.top_left.x, human.box.top_left.y-15),
                fontScale = 0.8,
                color=color,
                thickness=thickness,
                fontFace=cv2.LINE_AA
                )
    cv2.putText(image, area,
                (human.box.top_left.x + 200, human.box.top_left.y-15),
                fontScale = 0.8,
                color=color,
                thickness=thickness,
                fontFace=cv2.LINE_AA
                )

def visualize_item(item:Item, image, color:tuple, thickness, label):
    cv2.rectangle(image,
                    (item.box.top_left.x, item.box.top_left.y),
                    (item.box.bot_right.x, item.box.bot_right.y),
                    color=color,
                    thickness=thickness
                    )
    cv2.putText(image, label,
                (item.box.top_left.x, item.box.top_left.y-15),
                fontScale = 0.8,
                color=color,
                thickness=thickness,
                fontFace=cv2.LINE_AA
                )


def visual_object(input:dict, image):
    '''
    format input:
    {
        key: Human (or Item),
        value: {
            area: shelf/pay
            items: [item1, item2, item3, ...]
            is_pay: True/False
        },
    }
    '''
    # print(input)
    import ipdb; ipdb.set_trace()

    color = COLOR.green if input.values() == 'pay' else COLOR.blue
    visualize_human(input.keys(), 
                        image=image, 
                        color=color,
                        thickness=1,
                        label=classes[input.keys().id],
                    )

def visualize_payment(image, payment_storage):
    title = "****PAYMENT****"
    x_start_human = 10
    y_start_human = 30
    cv2.putText(image,title,(55,y_start_human),cv2.FONT_HERSHEY_TRIPLEX, 1, (111, 0, 5), 1)
    x_start_item = x_start_human + 30
    y_start_human += 30
    for human_id in payment_storage:
        cv2.putText(image,f"person {human_id}",(x_start_human,y_start_human),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (111, 0, 5), 1)
        y_start_item = y_start_human + 30
        for item_obj in payment_storage[human_id]:
            cls_id = item_obj.cls_id
            cv2.putText(image,f"+ {classes[cls_id]}",(x_start_item,y_start_item),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (111, 0, 5), 1)
            y_start_item += 30

        y_start_human += y_start_item






    

