from ultils.object import Image
from config.config_common import AREA, COLOR
# import numpy as np
# import cv2

def visual_area(image:Image):
    area_payment = image.has_area(AREA.payment)
    area_shelve = image.has_area(AREA.shelve)
    area_selection = image.has_area(AREA.selection)
    area_payment.draw(image.img, thickness=2, label='payment', color=COLOR.blue)
    area_shelve.draw(image.img, thickness=2, label='shelve', color=COLOR.green)
    area_selection.draw(image.img, thickness=2, label='selection', color=COLOR.red)
    

