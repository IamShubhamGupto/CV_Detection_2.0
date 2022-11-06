import torch
import cv2
import numpy as np
from pathlib import Path
import logging
from utils_.ProcessOffset import ProcessOffset
# logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.DEBUG)

class ProcessFrame:

    def __init__(self, camera_width, camera_height):
        self.process_offset_object = ProcessOffset(camera_width, camera_height)

    def process_frame(self, color_image, torch_model_object, detect_red):
        conf_thres = 0.25  # Confidence threshold
        # Get bounding boxes
        results = torch_model_object(color_image)

        # Post process bounding boxes
        #rows = results.pandas().xyxy[0].to_numpy()

        detections_rows = results.pandas().xyxy

        for i in range(len(detections_rows)):
            rows = detections_rows[i].to_numpy()

        # Go through all detections
        BLUE_MIN=np.array([100,150,0],np.uint8)
        BLUE_MAX=np.array([140,255,255],np.uint8)
        for i in range(len(rows)):
            # if len(rows):
            # Get the bounding box of the first object (most confident)
            x_min, y_min, x_max, y_max, conf, cls, label = rows[i]
            x_min = int(x_min)
            y_min = int(y_min)
            x_max = int(x_max)
            y_max = int(y_max)
            # Coordinate system is as follows:
            # 0,0 is the top left corner of the image
            # x is the horizontal axis
            # y is the vertical axis
            # x_max, y_max is the bottom right corner of the screen

            # (0,0) --------> (x_max, 0)
            # |               |
            # |               |
            # |               |
            # |               |
            # |               |
            # (0, y_max) ----> (x_max, y_max)
            # logger.debug("({},{}) \n\n\n                     ({},{})".format(
                    # x_min, y_min, x_max, y_max))

            # PROMPT 2 - display horizontal and vertical offset
            bbox = [x_min, y_min, x_max, y_max]
            roi = color_image[y_min:y_max, x_min:x_max]
            
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # BLUE_MIN = np.array([0, 0, 200], np.uint8) #minimum value of blue pixel in BGR order
            # BLUE_MAX = np.array([50, 50, 255], np.uint8) #maximum value of blue pixel in BGR order
            mask = cv2.inRange(hsv_roi, BLUE_MIN, BLUE_MAX)
            blue_probs = int((cv2.countNonZero(mask)/(roi.size/3))*100)
            # blue_probs = int(np.count_nonzero(mask==255)/(roi.shape[0]*roi.shape[1])) 
            logger.debug("BLUE PROBS = %d", blue_probs)
            if blue_probs >= 25:
            # logger.info("MASK = ",mask)
                x_rad_offset, y_rad_offset = self.process_offset_object.calculate_offset((x_max-x_min, y_max-y_min))
                color_image = self.write_bbx_frame(
                    color_image, bbox, label, conf)
        # Display the image
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

    def write_bbx_frame(self, color_image, bbxs, label, conf):
        # Display the bounding box
        x_min, y_min, x_max, y_max = bbxs
        cv2.rectangle(color_image, (x_min, y_min), 
            (x_max, y_max), (0, 255, 0), 2)  # Draw with green color

        # Display the label with the confidence
        label_conf = label + " " + str(conf)
        cv2.putText(color_image, label_conf, (x_min,y_min),
             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return color_image