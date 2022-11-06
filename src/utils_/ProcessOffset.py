import cv2
import numpy as np
from pathlib import Path
import logging
import math
logger = logging.getLogger(Path(__file__).stem)
# logging.basicConfig(level = logging.DEBUG)
logger.setLevel(logging.DEBUG)

class ProcessOffset:
    def __init__(self, camera_width, camera_height, HFOV = 69.4, VFOV = 42.5, DFOV = 77.0):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.HFOV = HFOV
        self.VFOV = VFOV
        self.DFOV = DFOV
        self.focal_length = (0.5*camera_width)/math.tan(0.5*math.radians(HFOV))
        self.x_center = camera_width//2
        self.y_center = camera_height//2

    def calculate_offset(self, roi_center_offsets):
        x_roi,y_roi = roi_center_offsets
        x_rad_offset = math.atan((self.x_center - x_roi)/self.focal_length)
        y_rad_offset = math.atan((self.y_center - y_roi)/self.focal_length)
        logger.debug("PROMPT2 x angle offset %f y angle offset %f", x_rad_offset, y_rad_offset)
        return (x_rad_offset, y_rad_offset)




    