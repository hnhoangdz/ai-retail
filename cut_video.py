import time
import cv2
import os


def get_clip(input_filename, output_filename,  start_sec, end_sec):
    # input and output videos are probably mp4
    vidcap = cv2.VideoCapture(input_filename)

    if (vidcap.isOpened() == False):
        print("Error reading cap file")
    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))

    size = (frame_width, frame_height)

    # math to find starting and ending frame number
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec*fps)
    end_frame = int(end_sec*fps)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
    
    # open video writer
    vidwrite = cv2.VideoWriter(output_filename, 
                               cv2.VideoWriter_fourcc(*'MJPG'),
                                10 , 
                               size)
    
    frame_count = start_frame
    print(frame_count)
    while True and (frame_count < end_frame):
        success, image = vidcap.read()
        if success == False:
            print("vcl")
            break
        print(frame_count)
        vidwrite.write(image)  # write frame into video
        success, image = vidcap.read()  # read frame from video
        frame_count+=1
    vidwrite.release()
    vidcap.release()

input_filename = "demo2802.avi"
output_filename = "cut_p1.avi"
get_clip(input_filename, output_filename, 185, 242)

# cap = cv2.VideoCapture(url)
# i=0
# counter = 0
# if (cap.isOpened() == False):
#     print("Error reading cap file")

# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# size = (frame_width, frame_height)

# result = cv2.VideoWriter('demo2802.avi',
#                          cv2.VideoWriter_fourcc(*'MJPG'),
#                          10, size)

# while(cap.isOpened()):
#     t0 = time.time()
#     ret, frame = cap.read()
#     if ret == False:
#         break
    
#     result.write(frame)

#     # if i == 
#     print(1/(time.time()-t0))
#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('s'):
#             break
# result.release()
# cap.release()
# cv2.destroyAllWindows()