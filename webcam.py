import cv2
import os

saved_folder = "data_process/cam_AI_2"
url = "rtsp://admin:123456aA@10.1.134.142:554/Streaming/channels/001"
cap = cv2.VideoCapture(url)
i=0
counter = 0
if (cap.isOpened() == False):
    print("Error reading cap file")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter('demo_2702_v7.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i > 150:
        result.write(frame)

    i+=1

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
            break
result.release()
cap.release()
cv2.destroyAllWindows()