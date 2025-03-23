import os
import yaml
from ultralytics import YOLO
from src.pipline.traning_pipline import TraningPipline

# model = YOLO('D:\CVProjects\Traffic-Saftey-Project\model (1).pt')
# print('model loaded')

# model(source=0,show=True,conf=0.4,save=True,stream=True)

obj = TraningPipline()
obj.run_pipline()


