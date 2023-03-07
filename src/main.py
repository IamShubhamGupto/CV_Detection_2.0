import os
import cv2
import numpy as np
from camera.Camera import Camera
from utils_.ProcessFrame import ProcessFrame
import time

from pathlib import Path
import logging
logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(Path(__file__).stem)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    # camera_object = Camera("camera_thread", 1, video_path='../example/vid.mp4')
    # camera_object.start()

    try:
        import camera.camera_stream
        video_stream = camera_stream.CameraVideoStream(2,use_tapi=True)
        print("INFO: using CameraVideoStream() threaded capture")
    except BaseException:
        print("INFO: CameraVideoStream() module not found")
        video_stream = cv2.VideoCapture()
    # torch_model_object = TorchModel("model_thread", 2, weights_path='./model_/pt_files/best.pt')
    # torch_model_object.start()
    # video_stream = camera_object.get_video_stream()
    logger.debug("Got video stream")
    camera_width = video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
    camera_height = video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
    process_frame_object = ProcessFrame("process_thread", 3, camera_width, camera_height)
    process_frame_object.start()
    # torch_model = torch_model_object.get_model()
    logger.debug("Loaded torch model")
    detect_red = True # RED | BLUE
    prev_frame_time = 0
    new_frame_time = 0
    while video_stream.isOpened():
        print('yes')
        try:
            ret, frame = video_stream.read()
        except:
            logger.error("Error getting frame")

        # Prompt 2 - display True
        new_frame_time = time.time()
        process_frame_object.process_frame(color_image=frame, detect_red=detect_red)

        if ret:
            key = cv2.waitKey(1)
            if key == 27:
                break
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        logger.info(f"FPS={str(int(fps))}")
        # cv2.putText(frame, str(int(fps)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    # camera_object.join()
    # torch_model_object.joisn()
    process_frame_object.join()

    
if __name__=='__main__':
    main()