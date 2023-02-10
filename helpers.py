import cv2
import torch
import numpy as np

def coco_to_yolo(xmin, ymin, xmax, ymax, org_w, org_h):
    xcb, ycb, wb, hb = (xmax+xmin)/2, (ymax+ymin)/2, xmax-xmin, ymax-ymin
    return xcb/org_w, ycb/org_h, wb/org_w, hb/org_h

def yolo_to_coco(x, y, w, h, org_w, org_h):
    cx, cy = x*org_w, y*org_h
    bw, bh = w*org_w, h*org_h
    xmin = cx - bw/2
    ymin = cy - bh/2
    xmax = cx + bw/2
    ymax = cy + bh/2
    return xmin, ymin, xmax, ymax

def kpts_to_box(points):
    points = np.array(points)
    xmin,ymin = np.min(points, 0)
    xmax,ymax = np.max(points, 0)
    return [xmin, ymin, xmax, ymax]

def draw_line(image, xf1, yf1, xf2, yf2, color=(0, 255, 0)):
    w = image.shape[1]
    h = image.shape[0]

    start_point = (int(w*xf1), int(h*yf1))
    end_point = (int(w*xf2), int(h*yf2))

    # Gets intercept
    slope = (yf2-yf1)/(xf2-xf1)
    b = yf1 - slope*xf1
    print("yf = " + str(round(slope, 3)) + "*xf + " + str(round(b, 3)))

    cv2.line(image, start_point, end_point, color, 4)
    return start_point, end_point

def draw_box(image, xf1, yf1, w_box, h_box):
    w = image.shape[1]
    h = image.shape[0]

    x1 = int(w*xf1)
    y1 = int(h*yf1)
    x2 = x1 + w_box
    y2 = y1 + h_box
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)
    return x1, y1, x2, y2

def load_model(weight_path, device):
    model = torch.load(weight_path, map_location=device)[
        'model'].float().eval()
    if torch.cuda.is_available():
        model.half().to(device)
    return model

def search_area(image, midx, midy):
    
    w = image.shape[1]
    h = image.shape[0]
    xf = midx/w
    yf = midy/h
    x1, x2 = 0.136, 0.75 
    
    y1 = 0.0*xf + 0.0
    y2 = -6.2*xf + 4.65
    y3 = -0.337*xf + 0.546
    y4 = 25.0*xf + -3.4
    
    # print(int(y1*h))
    # print(int(y2*h))
    # print(int(y3*h))
    # print(int(y4*h))
    
    # cv2.putText(image, "y1", (int(midx), int(y1*h)), cv2.FONT_HERSHEY_SIMPLEX, 2, (120,0,255), 2, cv2.LINE_AA)
    # cv2.putText(image, "y2", (int(midx), int(y2*h)), cv2.FONT_HERSHEY_SIMPLEX, 2, (120,0,255), 2, cv2.LINE_AA)
    # cv2.putText(image, "y3", (int(midx), int(y3*h)), cv2.FONT_HERSHEY_SIMPLEX, 2, (120,0,255), 2, cv2.LINE_AA)
    # cv2.putText(image, "y4", (int(midx), int(y4*h)), cv2.FONT_HERSHEY_SIMPLEX, 2, (120,0,255), 2, cv2.LINE_AA)
    
    if x1 <= xf <= x2:
        if y1 <= yf <= y3:
            return "shelf"
    return "out-shelf"

def which_area(image, midx, midy):

    w = image.shape[1]
    h = image.shape[0]
    xf = midx/w
    yf = midy/h
    print("xf = ", xf)
    # x sections
    x1, x2, x3, x4, x5, x6 = 0.10, 0.30, 0.35, 0.55, 0.65, 0.85
    # y=mx+b equations that separate each section
    y1 = 0.0*xf + 0.2  # Top-left line
    y2 = -0.444*xf + 0.294  # Top-middle line
    y3 = 2.75*xf + -0.025  # Left line
    y4 = -1.0*xf + 1.1  # Bottom line
    y5 = 1.0*xf + -0.2  # Middle Line
    
    print(int(y1*h))
    print(int(y2*h))
    print(int(y3*h))
    print(int(y4*h))
    print(int(y5*h))
    
    cv2.putText(image, "y1", (int(midx), int(y1*h)), cv2.FONT_HERSHEY_SIMPLEX, 2, (120,0,255), 2, cv2.LINE_AA)
    cv2.putText(image, "y2", (int(midx), int(y2*h)), cv2.FONT_HERSHEY_SIMPLEX, 2, (120,0,255), 2, cv2.LINE_AA)
    cv2.putText(image, "y3", (int(midx), int(y3*h)), cv2.FONT_HERSHEY_SIMPLEX, 2, (120,0,255), 2, cv2.LINE_AA)
    cv2.putText(image, "y4", (int(midx), int(y4*h)), cv2.FONT_HERSHEY_SIMPLEX, 2, (120,0,255), 2, cv2.LINE_AA)
    cv2.putText(image, "y5", (int(midx), int(y5*h)), cv2.FONT_HERSHEY_SIMPLEX, 2, (120,0,255), 2, cv2.LINE_AA)
    
    if xf <= x1:
        if yf <= y1:  # Top-left line
            area = "A2"
        else:
            area = "Register"
    elif xf > x1 and xf <= x2:
        if yf <= y2:  # Top-middle line
            area = "A2"
        elif yf <= y3:  # Left line
            area = "A3"
        else:
            area = "Register"
    elif xf > x2 and xf <= x3:
        if yf <= y2:  # Top-middle line
            area = "A2"
        elif yf <= y4:  # Bottom line
            area = "A3"
        else:
            area = "Entrance"
    elif xf > x3 and xf <= x4:
        if yf <= y2:  # Top-middle line
            area = "A2"
        elif yf <= y5:  # Middle Line
            area = "A1"
        elif yf <= y4:  # Bottom line
            area = "A3"
        else:
            area = "Entrance"
    elif xf > x4 and xf <= x5:
        if yf <= y5:  # Middle Line
            area = "A1"
        elif yf <= y4:  # Bottom line
            area = "A3"
        else:
            area = "Entrance"
    elif xf > x5 and xf <= x6:
        if yf <= y4:  # Bottom line
            area = "A1"
        else:
            area = "Entrance"

    return area

def writes_area_text(image, text, xf1, yf1):
    w = image.shape[1]
    h = image.shape[0]

    start_point = (int(w*xf1), int(h*yf1))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.8
    color = (255, 100, 100)
    thickness = 2

    # Draws background text
    cv2.putText(image, text,
                start_point, font, fontScale, (0, 0, 0), thickness+3)

    # Draws foreground text
    cv2.putText(image, text,
                start_point, font, fontScale, color, thickness)

def bbox_to_center(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return int(x1 + w/2), int(y1 + h/2)
