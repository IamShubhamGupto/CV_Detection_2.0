import cv2
import numpy as np
import logging
from pathlib import Path
logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(level=logging.DEBUG)

class Camera(object):
    def __init__(self, camera_id=0):
        self.camera_id=camera_id
        self.video_stream = cv2.VideoCapture(camera_id)
        logger.debug('initialized Camera class')

    def __del__(self):
        self.video_stream.release()

    def get_video_stream(self):
        logger.debug('getting video stream')
        return self.video_stream

    def get_camera_id(self):
        logger.debug('getting camera id %d'.format(self.get_camera_id))
        return self.get_camera_id
    
    
    

