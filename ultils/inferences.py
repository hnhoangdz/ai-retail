import sys
sys.path.insert(0, './yolov7')
import numpy as np
import torch

from detection.yolov7.models.experimental import attempt_load
from detection.yolov7.utils.datasets import LoadImages
from detection.yolov7.utils.general import check_img_size, non_max_suppression, apply_classifier, set_logging, scale_coords
from detection.yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized
from detection.yolov7.utils.datasets import letterbox

def load_model(weight_path, device):
    model = torch.load(weight_path, map_location=device)['model'].float().fuse().eval()
    if torch.cuda.is_available():
        model.to(device)
    return model

def yolov7_inference(detector, image, classes=None, conf_thresh=0.45, iou_thresh=0.3, device="0"):
    # half = False
    # if str(device) != "cpu":
    half = True
    img = image.copy()
    img = letterbox(img, new_shape=(640, 640))[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    t1 = time_synchronized()
    with torch.no_grad():
        pred = detector(img, augment=False)[0]  # list: bz * [ (#obj, 6)]
    # Apply NMS
    pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes, agnostic=False)
    t2 = time_synchronized()
    fps = 1/(t2-t1)
    
    # Get final results
    results_det = pred[0]
    if results_det is not None and len(results_det):
        results_det[:, :4] = scale_coords(img.shape[2:], results_det[:, :4], image.shape).round()
        return results_det, fps
    else: 
        return torch.zeros((0, 5)), fps  


