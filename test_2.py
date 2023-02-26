from ultils.object import Image
from config.config_common import AREA, COLOR
import cv2

image = Image(id=None, img='data/sample.png', ratio=1)
area_payment = image.has_area(AREA.payment)
area_shelve = image.has_area(AREA.shelve)
area_selection = image.has_area(AREA.selection)
area_payment.draw(image.img, thickness=2, label='payment', color=COLOR.blue)
area_shelve.draw(image.img, thickness=2, label='shelve', color=COLOR.green)
area_selection.draw(image.img, thickness=2, label='selection', color=COLOR.red)

cv2.imwrite("./draw_area.png", img=image.img)
# import numpy as np
# import cv2
# img = np.zeros((512, 512,3), dtype = "uint8")
# rectangle = np.array([[10,5],[10,225],[50,225],[50,5]], np.int32)
# triangle = np.array([[60,10],[60,200],[150,100]], np.int32)
# import ipdb; ipdb.set_trace()  

# rectangleImage =cv2.polylines(img, [rectangle], False, (0,255,0), thickness=3)
# triangleImage =cv2.polylines(img, [triangle], False, (0,0,255), thickness=3)
# cv2.imshow('Shapes', rectangleImage)
# cv2.imshow('Shapes', triangleImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
