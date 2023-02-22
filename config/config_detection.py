from config.config_common import CLASSES
from keypoint import keypoint_definition
class YoloConfig:
    # weight
    weight_path = '/home/ubuntu/Desktop/research/yolov7/runs/train/train_detect_v33/weights/best.pt'

    # threshold
    conf_thres = 0.5
    iou_thres = 0.3
    agnostic = False

    # classes
    classes = [CLASSES.person, CLASSES.oreo, CLASSES.chinsu, CLASSES.coca, CLASSES.fanta]

    # device
    device = 'cuda:0'

class PoseConfig:
    checkpoint = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"

    config = 'tracking/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py'
    
    bbox_thres = 0.3
    formats = 'xyxy',
    return_heatmap = False
    outputs = None
    use_oks = True
    tracking_thr = 0.3
    pose_define = keypoint_definition
    pose_left_hand = [i for i in range(91, 112, 1)] + [9]
    pose_right_hand = [i for i in range(112, 133, 1)] + [10]
    pose_left_leg = [15, 17, 18, 19]
    pose_left_leg = [16, 20, 21, 22]



    