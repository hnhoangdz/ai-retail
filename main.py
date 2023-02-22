
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
from typing import List, Dict
from ultils.logger import set_logger
                                
def make_parser():
    parser = argparse.ArgumentParser("Retail Store Demo!")
    
    # Detection
    parser.add_argument("--weights_path", type=str, default="./weights/best.pt")
    # parser.add_argument("--weights_item", type=str, default="/home/hoangdinhhuy/hoangdinhhuy/VTI/retail_store/yolov7/trained_models/item_detector.pt")
    parser.add_argument("--conf_thresh", type=float, default=0.45, help="confidence threshold object detection")
    parser.add_argument("--iou_thresh", type=float, default=0.3, help="iou threshold object detection")
    parser.add_argument("--video", type=str, default="", help="input video link")

    # Input and ouput
    parser.add_argument("--input_path", type=str, default="/home/hoangdinhhuy/hoangdinhhuy/VTI/retail_store/output.avi", help="input video")
    parser.add_argument("--save_path", type=str, default="./results", help="output folder")
    parser.add_argument("--fourcc", type=str, default="mp4v", help="output video codec (verify ffmpeg support)")
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # Camera
    parser.add_argument("--display", default=True, action="store_true")
    parser.add_argument("--cam", action="store", type=str, default="-1")

    # Tracking args
    parser.add_argument("--input_size", type=tuple, default=(800, 1440), help="input size image in tracker")
    parser.add_argument("--track_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.45, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.0,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def main():
    args = make_parser().parse_args()



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Devide:{device}")
    args = get_parser().parse_args()
    
    # Detection
    detector = load_model(args.weights_path, device)
    # item_detector = load_model(args.weights_item, device)
        
    # Tracking
    tracker = BYTETracker(args)
    with VideoRetailStore(args) as vidRS:
        vidRS.run()




