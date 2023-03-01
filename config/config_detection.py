from config.config_common import CLASSES
from keypoint import keypoint_definition
class Yolov7Config:
    # weight
    weight_path = '/home/ubuntu/Desktop/research/yolov7/runs/train/haha9/weights/best.pt'

    # threshold
    conf_thres = 0.8
    iou_thres = 0.45
    agnostic = False

    # classes
    classes = [CLASSES.g7, CLASSES.nabaty, CLASSES.hop_xanh, CLASSES.haohao, CLASSES.mi_do, CLASSES.nescafe, CLASSES.mi_vang, CLASSES.cafe_trang]

    # device
    device = 'cuda:0'

# ronaldod
class Yolov8Config:
    weight_path = '/home/ubuntu/Desktop/research/yolov7/yolov7.pt'

    # threshold
    conf_thres = 0.6
    iou_thres = 0.45
    agnostic_nms = False

    # classes
    classes = [CLASSES.g7, CLASSES.nabaty, CLASSES.hop_xanh, CLASSES.haohao, CLASSES.mi_do, CLASSES.nescafe, CLASSES.mi_vang, CLASSES.cafe_trang]

    # device
    device = 'cuda:0'

    retina_masks=True,


class PoseConfig:
    checkpoint = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"

    config = 'tracking/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py'
    
    bbox_thres = 0.8
    formats = 'xyxy',
    return_heatmap = False
    ouputs = None
    use_oks = True
    tracking_thr = 0.8
    id_pose_lefthand = [i for i in range(91, 112, 1)] + [9]
    id_pose_righthand = [i for i in range(112, 133, 1)] + [10]
    id_pose_leftleg = [15, 17, 18, 19]
    id_pose_rightleg = [16, 20, 21, 22]
    keypoint_definition = keypoint_definition
    device='cuda:0'



    