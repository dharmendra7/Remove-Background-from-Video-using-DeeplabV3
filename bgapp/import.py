import cv2

cap = cv2.VideoCapture("media/videos/69b1f2c7-b06.mov")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(length)
