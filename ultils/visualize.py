from yolov7.object import Human, Item, CLASSES
from helpers import draw_line
import cv2
from typing import Dict, List
from cfg import COLOR, classes


def visualize_human(human:Human, image, color:tuple, thickness, label):
    # visual Box
    const_edge_len = 10
    # top left
    cv2.line(image, 
            (human.box.top_left.x, human.box.top_left.y),
            (human.box.top_left.x + const_edge_len, human.box.top_left.y + const_edge_len),
            color, thickness) 
    # bottom_right
    cv2.line(image, 
            (human.box.bot_right.x, human.box.bot_right.y),
            (human.box.bot_right.x - const_edge_len, human.box.bot_right.y - const_edge_len),
            color, thickness) 
    # visual Label
    cv2.putText(image, 
                text=label
                (human.box.top_left.x, human.box.top_left.y-5),
                font=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=color,
                thickness=thickness
                )

def visualize_item(item:Item, image, color:tuple, thickness, label):
    cv2.rectangle(image,
                    (item.box.top_left.x, item.box.top_left.y),
                    (item.box.bot_right.x, item.box.bot_right.y),
                    color=color,
                    thickness=thickness
                    )
    cv2.putText(image, 
                text=label
                (item.box.top_left.x, item.box.top_left.y-5),
                font=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=color,
                thickness=thickness
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
    # import ipdb; ipdb.set_trace()
    if isinstance(input.keys(), Human):
        color = COLOR.green if input.values() == 'pay' else COLOR.blue
        visualize_human(input.keys(), 
                            image=image, 
                            color=color,
                            thickness=1,
                            label=classes[input.keys().id],
                        )

    # else:
    #     import ipdb; ipdb.set_trace()
    #     color = COLOR.magenta if input.values() == 'pay' else COLOR.yellow
    #     visualize_item(input.keys(), 
    #                         image=image, 
    #                         color=color,
    #                         thickness=1,
    #                         label=classes[input.keys().id],
    #                     )






    

