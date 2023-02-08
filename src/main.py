import os
import cv2
import numpy as np
# from camera.Camera import Camera
from camera.WebcamVideoStream import WebcamVideoStream
from model_.TorchModel import TorchModel
from utils_.ProcessFrame import ProcessFrame

from pathlib import Path
import logging
logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(Path(__file__).stem)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    # camera_object = Camera(video_path='../example/vid.mp4')
    torch_model_object = TorchModel(weights_path='./model_/pt_files/best.pt')
    # video_stream = camera_object.get_video_stream()
    video_stream = WebcamVideoStream(src='../example/vid.mp4').start()
    logger.debug("Got video stream")
    camera_width = video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
    camera_height = video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
    process_frame_object = ProcessFrame(camera_width, camera_height)
    torch_model = torch_model_object.get_model()
    logger.debug("Loaded torch model")
    detect_red = True # RED | BLUE
    # org
    org = (00, 185)
    # fontScale
    fontScale = 1
    
    # Red color in BGR
    color = (0, 120, 255)
    # Line thickness of 2 px
    thickness = 2
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX 
    while video_stream.isOpened():
        # cv2 index 5 is fps
        # logger.info("FPS %f",video_stream.get(5))
        try:
            # ret, frame = video_stream.read()
            frame = video_stream.read()
        except Exception as e:
            logger.error("Error getting frame %s", e)
            continue

        # Prompt 2 - display True

        process_frame_object.process_frame(color_image=frame, torch_model_object=torch_model, detect_red=True)
        if cv2.waitKey(1) == 27:
            break
    video_stream.stop()
    video_stream.__del__
if __name__=='__main__':
    main()
    