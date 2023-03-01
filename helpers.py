import cv2
import torch
import numpy as np
from cfg import *

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

    cv2.line(image, start_point, end_point, color, 1)

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
    model = torch.load(weight_path, map_location=device)['model'].float().fuse().eval()
    if torch.cuda.is_available():
        model.to(device)
    return model

def search_area(w, h, midx, midy):
    
    xf = midx/w
    yf = midy/h
    
    # Shelf x
    x1s, x2s = shelf_area[0][0], shelf_area[1][0] 
    
    # In-out x 
    x1io, x2io = in_out_area[0][0], in_out_area[1][0]
    
    # Payment x
    x1p, x2p, x3p, x4p = payment_area[0][0], payment_area[1][0], payment_area[2][0], payment_area[1][0]
    
    # In-out
    yfio = 2.267*xf + -1.173
    # Shelf
    yfs = 4.46*xf + -1.66

    # Payment
    yfp1 = -0.22*xf + 0.802
    yfp2 = 2.935*xf + -1.564
    yfp3 = -0.538*xf + 1.308
    yfp4 = 3.214*xf + -1.259

    # cv2.circle(image, (int(midx), int(midy)), 5, (0,255,0), 5, cv2.INTER_LINEAR)
    # cv2.circle(image, (int(midx), int(yfio*h)), 5, (255,0,0), 5, cv2.INTER_LINEAR)
    # cv2.circle(image, (int(midx), int(yfs*h)), 5, (255,0,0), 5, cv2.INTER_LINEAR)
    # cv2.circle(image, (int(midx), int(yfp1*h)), 5, (255,0,0), 5, cv2.INTER_LINEAR)
    # cv2.circle(image, (int(midx), int(yfp2*h)), 5, (255,0,0), 5, cv2.INTER_LINEAR)
    # cv2.circle(image, (int(midx), int(yfp3*h)), 5, (255,0,0), 5, cv2.INTER_LINEAR)
    # cv2.circle(image, (int(midx), int(yfp4*h)), 5, (255,0,0), 5, cv2.INTER_LINEAR)

    area = ""
    if x1s < xf < x2s:
        if yf > yfs:
            area = "shelf"
        else:
            area = "attend"
    elif xf > x2s:
        if yf < yfs:
            area = "attend"
    
    if xf < x1io:
        if yf < yfio:
            area = "outside"
    elif xf < x2io:
        if yf < yfio:
            area = "outside"
        else:
            area = "attend"
    else:
        if yf > yfio:
            area = "attend"
        else:
            area = "outside"
        
    if x1p < xf < x2p:
        if yfp1 < yf:
            if (yf < yfp4) or (yf < yfp3):
                area = "payment"
            
    elif x4p < xf < x3p:
        if yfp1 < yf:
            if (yf < yfp3) or (yfp2 < yf < yfp3):
                area = "payment"

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

def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou


def get_max_iou(pred_boxes, gt_box):
    """
    calculate the iou multiple pred_boxes and 1 gt_box (the same one)
    pred_boxes: multiple predict  boxes coordinate
    gt_box: ground truth bounding  box coordinate
    return: the max overlaps about pred_boxes and gt_box
    """
    # 1. calculate the inters coordinate
    if pred_boxes.shape[0] > 0:
        ixmin = np.maximum(pred_boxes[:, 0], gt_box[0])
        ixmax = np.minimum(pred_boxes[:, 2], gt_box[2])
        iymin = np.maximum(pred_boxes[:, 1], gt_box[1])
        iymax = np.minimum(pred_boxes[:, 3], gt_box[3])

        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

    # 2.calculate the area of inters
        inters = iw * ih

    # 3.calculate the area of union
        uni = ((pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) +
               (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
               inters)

    # 4.calculate the overlaps and find the max overlap ,the max overlaps index for pred_box
        iou = inters / uni
        iou_max = np.max(iou)
        nmax = np.argmax(iou)
        return iou, iou_max, nmax
