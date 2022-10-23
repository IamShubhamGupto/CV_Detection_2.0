import torch
import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(level=logging.DEBUG)

class ProcessFrame(object):

    def process_frame(self, color_image, torch_model_object, display=False):
        conf_thres = 0.25  # Confidence threshold
        # Get bounding boxes
        results = torch_model_object(color_image)

        # Post process bounding boxes
        #rows = results.pandas().xyxy[0].to_numpy()

        detections_rows = results.pandas().xyxy

        for i in range(len(detections_rows)):
            rows = detections_rows[i].to_numpy()

        # Go through all detections

        for i in range(len(rows)):
            # if len(rows):
            # Get the bounding box of the first object (most confident)
            x_min, y_min, x_max, y_max, conf, cls, label = rows[i]

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
            logger.debug("({},{}) \n\n\n                     ({},{})".format(
                    x_min, y_min, x_max, y_max))

            if display:
                bbox = [x_min, y_min, x_max, y_max]
                color_image = self.write_bbx_frame(
                    color_image, bbox, label, conf)
        # Display the image
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

    def write_bbx_frame(self, color_image, bbxs, label, conf):
        # Display the bounding box
        x_min, y_min, x_max, y_max = bbxs
        cv2.rectangle(color_image, (int(x_min), int(y_min)), (int(
            x_max), int(y_max)), (0, 255, 0), 2)  # Draw with green color

        # Display the label with the confidence
        label_conf = label + " " + str(conf)
        cv2.putText(color_image, label_conf, (int(x_min), int(
            y_min)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return color_image