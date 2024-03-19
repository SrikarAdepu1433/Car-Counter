# Car Counter using YOLO Object Detection and SORT Algorithm

This repository contains Python code for a car counting system using the YOLO (You Only Look Once) object detection model and the SORT (Simple Online and Realtime Tracking) algorithm. The system detects cars in a video stream, tracks their movement, and counts them as they pass through a designated region.

## Dependencies
- `numpy`
- `cv2` (OpenCV)
- `cvzone`
- `sort` (Simple Online and Realtime Tracking algorithm)
- `ultralytics` (for YOLO object detection)

## Usage
1. Clone the repository.
2. Install the dependencies using pip.
3. Download the YOLO weights (yolov8l.pt) and place them in the specified directory.
4. Run the Python script (`car_counter.py`) with a video file as input (e.g., `cars.mp4`).
5. View the output with car counts displayed on the screen.

## Description
- The code first imports necessary libraries including OpenCV for image processing, YOLO for object detection, and SORT for object tracking.
- It loads the video file and sets up parameters for the YOLO model and SORT tracker.
- The YOLO model is applied to detect objects in each frame of the video.
- Cars are filtered from the detected objects based on predefined class labels.
- The SORT algorithm is used to track the detected cars across frames.
- A specified region in the video frame is defined to count cars passing through it.
- The count of cars passing through the region is displayed on the screen.

## Note
- This code assumes that the YOLO model weights (`yolov8l.pt`) are available in the specified directory.
- Adjustments to parameters such as confidence thresholds and region of interest can be made in the code as needed.
