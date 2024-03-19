import math
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
from sort import *

video = cv2.VideoCapture("cars.mp4")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
mask = cv2.imread("mask1.png")
# Tracking  the classes
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [280, 350, 673, 350]
model = YOLO('../YOLO-Weights/yolov8l.pt')
totalcount = []
while True:
    success, img = video.read()
    imgRegion = cv2.bitwise_and(img, mask)
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    result = model(imgRegion, stream=True)
    detections = np.empty((0, 5))
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1

            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            CurrentCLASS = classNames[cls]
            if CurrentCLASS == "car":
                    #or CurrentCLASS == "motorbike" or CurrentCLASS == "truck" or CurrentCLASS == "bus" or CurrentCLASS == "bicycle" and conf > 0.3):
                #cvzone.putTextRect(img, f"{CurrentCLASS} ", (max(0, x1), max(35, y1)), scale=1, thickness=0)
                #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 250, 0), 2)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack([detections, currentArray])

        resultTracker = tracker.update(detections)
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 3)
        for result in resultTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
            w, h = x2-x1, y2-y1
            # YOLO DETECTIONS
            #cv2.rectangle(img, (x1, y1), (x2, y2), color=(250, 0, 0), thickness=2)
            #cvzone.cornerRect(img, (x1, y1, w, h), l=9,colorR=(250, 0, 0))


            # code for id printers
            cvzone.putTextRect(img, f"{int(id)}  ", (max(0, x1), max(35, y1)), scale=1, thickness=0)
            # code for centers dot that moves along with the classes
            cx, cy = x1+w//2, y1+h//2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            # code for region where the object should be detected and count cars

            if limits[0] < cx < limits[2] and limits[1]-13 < cy < limits[1]+13:
                if totalcount.count(id) == 0:
                    totalcount.append(id)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 3)
            #cvzone.putTextRect(img, f" count: {len(totalcount)} ", (50, 50), scale=2, thickness=2)
        # printing of the car counters on the screen
        cvzone.putTextRect(img, str(len(totalcount)), (255, 100), 2, 3, (0, 0, 0), (50, 50, 250), cv2.FONT_HERSHEY_PLAIN,border=None)
        cv2.imshow("Image", img)
        #cv2.imshow("imgRegion", imgRegion)
        cv2.waitKey(1)