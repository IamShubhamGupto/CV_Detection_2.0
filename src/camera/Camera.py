import cv2
import numpy as np
import logging
from pathlib import Path
# logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.DEBUG)


class Camera:
    def __init__(self, camera_id=0, video_path=None):
        if video_path is None:
            self.camera_id=camera_id
            self.video_stream = cv2.VideoCapture(camera_id)
        else:
            logger.debug('opening video %s',video_path)
            self.camera_id = -1
            self.video_stream = cv2.VideoCapture(video_path)
        logger.debug('initialized Camera class')

    def __del__(self):
        self.video_stream.release()

    def get_video_stream(self):
        logger.debug('getting video stream')
        return self.video_stream

    def get_camera_id(self):
        logger.debug('getting camera id %d'.format(self.get_camera_id))
        return self.camera_id
    
    def get_camera_fps(self):
        return self.video_stream.get(5)
    
    
    

