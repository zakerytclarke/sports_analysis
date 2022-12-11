import torch
import matplotlib as plt
# Loading in yolov5s - you can switch to larger models such as yolov5m or yolov5l, or smaller such as yolov5n
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
img = 'https://i.ytimg.com/vi/q71MCWAEfL8/maxresdefault.jpg'  # or file, Path, PIL, OpenCV, numpy, list
results = model(img)
