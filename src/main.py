import os
import cv2
import numpy as np
from camera.Camera import Camera
from model.TorchModel import TorchModel
from utils.ProcessFrame import ProcessFrame
from pathlib import Path
import logging

logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(level=logging.DEBUG)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    camera_object = Camera(0)
    torch_model_object = TorchModel(weights_path='./model/pt_files/best.pt')
    process_frame_object = ProcessFrame()
    video_stream = camera_object.get_video_stream()
    torch_model = torch_model_object.get_model()
    while True:
        try:
            ret, frame = video_stream.read()
        except:
            logger.error("Error getting frame")

        process_frame_object.process_frame(color_image=frame, torch_model_object=torch_model, display=True)

        if ret:
            key = cv2.waitKey(1)
            if key == 27:
                break
if __name__=='__main__':
    main()