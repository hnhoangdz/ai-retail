import cv2
import matplotlib.pyplot as plt
from helpers import which_area, draw_line
import os
import glob

# for img_path in glob.glob("/home/hoangdinhhuy/hoangdinhhuy/VTI/retail_store/food/*.jpg"):
#     img = cv2.imread(img_path)
#     base_name = os.path.basename(img_path).replace(".txt", "")
#     cv2.imshow(base_name, img)
#     cv2.waitKey()

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
        else:
            return "out-shelf"
    # cv2.putText(image, "y5", (int(midx), int(y5*h)), cv2.FONT_HERSHEY_SIMPLEX, 2, (120,0,255), 2, cv2.LINE_AA)
        # pass

img = cv2.imread("c.jpg")
draw_line(img, 0.136, 0.01, 0.75, 0.01)
draw_line(img, 0.75, 0.01, 0.7, 0.31)
draw_line(img, 0.136, 0.5, 0.7, 0.31)
draw_line(img, 0.136, 0.01, 0.156, 0.5)
cx,cy = 694,15
cv2.circle(img, (cx, cy), 2, (0,255,0), 2, cv2.LINE_AA)
search_area(img, cx, cy)
cv2.imshow("", img)
cv2.waitKey(0)

# with open("CARDS_COURTYARD_B_T_frame_0295_jpg.rf.479a6987460ca993ce070ce5277c6ed9.txt", "r") as f:
#     data = f.readlines()
#     for i in range(len(data)):
#         s = data[i]
#         print(s)
#         s = s.replace(s[0], "0")
#         print(s)
#         # print(data[i][0])
#         # data[i].replace(data[i][0], "0")
#         # print(data[i])
# print(data)