import cv2
import numpy as np
import cfg

cap = cv2.VideoCapture('rtsp://admin:123456aA@10.1.134.142:554/Streaming/channels/001')
if (cap.isOpened()== False):
	print("Error opening video file")

payment_frame = np.zeros([1080, 1920//4, 3], dtype=np.uint8)
payment_frame[:, :] = [255, 255, 255]

while(cap.isOpened()):

	ret, frame = cap.read()
	
	if ret == True:

		frame = cv2.resize(frame, (1920, 1080))
		# Title Payment
		retail_frame = cv2.resize(frame, (1920*3//4, 1080))
		print(retail_frame.shape[:2])
		final_frame = np.concatenate((retail_frame, payment_frame), 1)

		# Display the resulting frame
		cv2.imshow('Frame', final_frame)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
	else:
		break

cap.release()
cv2.destroyAllWindows()

