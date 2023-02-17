import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def load_model(weight_path):
#     model = torch.load(weight_path, map_location=device)['model']
#     model.float().eval()
    
#     if torch.cuda.is_available():
#         model.half().to(device)
    
#     return model

# def run_inference(img, model):
#     # Resize and pad image
#     img = letterbox(img, 960, stride=64, auto=True)[0]
#     # Apply transforms
#     img = transforms.ToTensor()(img)
#     # Turn image into batch
#     img = img.unsqueeze(0)
#     # Predict
#     output, _ = model(img)
#     return output, img

# def visualize_output(output, image, model):
#     output = non_max_suppression_kpt(output, 
#                                      0.25, # Confidence Threshold
#                                      0.65, # IoU Threshold
#                                      nc=model.yaml['nc'], # Number of Classes
#                                      nkpt=model.yaml['nkpt'], # Number of Keypoints
#                                      kpt_label=True)
#     with torch.no_grad():
#         output = output_to_keypoint(output)
#     nimg = image[0].permute(1, 2, 0) * 255
#     nimg = nimg.cpu().numpy().astype(np.uint8)
#     nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
#     for idx in range(output.shape[0]):
#         plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
#     return nimg

def get_keypoints(img, results_det, margin=15, det_conf=0.25, track_conf=0.25):
    final_kpts = []
    height, width = img.shape[:2]
    print(len(results_det))
    c = 0
    for i, (xmin, ymin, xmax, ymax, conf, cls) in enumerate(results_det):
        
        per_kpts = []
        mg_ymin = max(0, int(ymin)-margin)
        mg_ymax = min(height, int(ymax)+margin)
        mg_xmin = max(0, int(xmin)-margin)
        mg_xmax = min(width, int(xmax)+margin)
        
        cropped_img = img[mg_ymin:mg_ymax, mg_xmin:mg_xmax]
        # cv2.imwrite(f"a_{i}.jpg", cropped_img)
        h, w, _ = cropped_img.shape
        with mp_pose.Pose(min_detection_confidence=det_conf, min_tracking_confidence=track_conf) as pose:
            results = pose.process(cropped_img)
            if results.pose_landmarks:
                c += 1
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    per_kpts.append([cx, cy])
        final_kpts.append(per_kpts)
    # if c != len(results_det):
    #     exit()
    return final_kpts

def get_hands_kpts(body_kpts):
    hands_kpts = []
    for kpts in body_kpts:
        tmp_hands_kpts = {"left": [], "right": []}
        for i, pts in enumerate(kpts):
            if i == 16 or i == 18 or i == 20 or i == 22:
                tmp_hands_kpts["right"].append(pts)
            elif i == 15 or i == 17 or i == 19 or i == 21:
                tmp_hands_kpts["left"].append(pts)
        hands_kpts.append([tmp_hands_kpts])
    return hands_kpts

def kpts_to_box(points):
    points = np.array(points)
    xmin,ymin = np.min(points, 0)
    xmax,ymax = np.max(points, 0)
    return [xmin, ymin, xmax, ymax]

def get_hands_box(hands_kpts):
    hands_box = []
    for per_hands in hands_kpts:
        left_hands_kpts = per_hands[0]["left"]
        right_hands_kpts = per_hands[0]["right"]
        
        if len(left_hands_kpts) == 0 or len(right_hands_kpts) == 0:
            continue
        
        left_hands_box = kpts_to_box(left_hands_kpts)
        right_hands_box = kpts_to_box(right_hands_kpts)
        hands_box.append([left_hands_box, right_hands_box])
    return hands_box

if __name__ == "__main__":
    pass
    # model = load_model("/home/hoangdinhhuy/hoangdinhhuy/VTI/retail_store/yolov7/trained_models/pose.pt")
    # video = cv2.VideoCapture("/home/hoangdinhhuy/hoangdinhhuy/VTI/retail_store/video_duy.mp4")

    # while True:
    #     t0 = time.time()
    #     ret, frame = video.read()
        
    #     # img = cv2.imread("/home/hoangdinhhuy/hoangdinhhuy/VTI/retail_store/1.jpg")

    #     output, image = run_inference(frame, model) # Bryan Reyes on Unsplash
    #     frame = visualize_output(output, image, model)
    #     print(1/(time.time() - t0))
    #     cv2.imshow("", frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # # After the loop release the cap object
    # video.release()
    # # Destroy all the windows
    # cv2.destroyAllWindows()