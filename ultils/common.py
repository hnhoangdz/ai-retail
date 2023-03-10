from ultils.object import Frame, Point
from config.config_common import AREA, COLOR
import tkinter as tk
# import numpy as np
# import cv2

def visual_area(image:Frame):
    area_payment = image.has_area(AREA.payment)
    line_shelve = image.has_area(AREA.line_shelve)
    # area_shelve = image.has_area(AREA.shelve)
    area_selection = image.has_area(AREA.selection)
    area_payment.draw(image.img, thickness=2, label='payment', color=COLOR.blue)
    # area_shelve.draw(image.img, thickness=2, label='shelve', color=COLOR.green)
    line_shelve.draw(image.img, thickness=2, label='line shelve', color=COLOR.green)
    area_selection.draw(image.img, thickness=2, label='selection', color=COLOR.red)

def convert2Real(x, y, img_wid:float, img_hght:float) -> Point:
    x = x * img_wid
    y = y * img_hght
    return (x, y)
    

def get_monitor_size():
    root = tk.Tk()

    width_px = root.winfo_screenwidth()
    height_px = root.winfo_screenheight()
    width_mm = root.winfo_screenmmwidth()
    height_mm = root.winfo_screenmmheight()
    # 2.54 cm = in
    width_in = width_mm / 25.4
    height_in = height_mm / 25.4
    width_dpi = width_px/width_in
    height_dpi = height_px/height_in
    return width_px, height_px, width_mm, height_mm, width_in, height_in, width_dpi, height_dpi