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

def draw_line(image, xf1, yf1, xf2, yf2, color=(0, 0, 125)):
    w = image.shape[1]
    h = image.shape[0]

    start_point = (int(w*xf1), int(h*yf1))
    end_point = (int(w*xf2), int(h*yf2))

    slope = (yf2-yf1)/(xf2-xf1)
    b = yf1 - slope*xf1
    # print("yf = " + str(round(slope, 3)) + "*xf + " + str(round(b, 3)))

    cv2.line(image, start_point, end_point, color, 4)

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
    model = torch.load(weight_path, map_location=device)['model'].float().eval()
    if torch.cuda.is_available():
        model.half().to(device)
    return model

def search_area(image, midx, midy):
    
    w = image.shape[1]
    h = image.shape[0]
    xf = midx/w
    yf = midy/h
    
    x1s, x2s, x3s, x4s = 0.2, 0.29, 0.39, 0.44
    x1aa, x2aa, x3aa, x4aa = 0.14, 0.4, 0.58, 0.73 
    
    yf1s = -1.211*xf + 0.672
    yf2s = 4.0*xf + -1.36
    yf3s = -1.6*xf + 1.104
    yf4s = 2.333*xf + -0.037
    
    yf1aa = -1.269*xf + 0.678
    yf2aa = 0.485*xf + -0.024
    yf3aa = -4.267*xf + 3.445
    yf4aa = 1.068*xf + 0.35
    
    cv2.circle(image, (int(midx), int(midy)), 5, (255,0,0), 5, cv2.INTER_LINEAR)
    cv2.circle(image, (int(midx), int(yf1s*h)), 5, (255,0,0), 5, cv2.INTER_LINEAR)
    # cv2.circle(image, (int(midx), int(yf2s*h)), 5, (255,0,0), 5, cv2.INTER_LINEAR)
    # cv2.circle(image, (int(midx), int(yf3s*h)), 5, (255,0,0), 5, cv2.INTER_LINEAR)
    # cv2.circle(image, (int(midx), int(yf4s*h)), 5, (255,0,0), 5, cv2.INTER_LINEAR)
    # cv2.circle(image, (int(midx), int(yf1aa*h)), 5, (255,0,0), 5, cv2.INTER_LINEAR)
    # cv2.circle(image, (int(midx), int(yf2aa*h)), 5, (255,0,0), 5, cv2.INTER_LINEAR)
    # cv2.circle(image, (int(midx), int(yf3aa*h)), 5, (255,0,0), 5, cv2.INTER_LINEAR)
    # cv2.circle(image, (int(midx), int(yf4aa*h)), 5, (255,0,0), 5, cv2.INTER_LINEAR)
    
    if x1aa <= xf <= x3aa:
        if x1s <= xf <= x3s:
            if yf < yf1s:
                return "outside"
            elif yf2s <= yf <= yf4s:
                return "shelf"
        if yf2aa <= yf <= yf4aa:
            return "attend"
    elif xf > x3aa:
        if yf > yf3aa:
            return "payment"
    return "outside"

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

# def 
