import torch
import cv2
import numpy as np
from pathlib import Path
import logging
# from utils_.Tracker import Tracker
from utils_.Sort import *
import matplotlib.pyplot as plt
# logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.DEBUG)

class ProcessFrame:
    def __init__(self, camera_width, camera_height, HFOV = 69.4, VFOV = 42.5, DFOV = 77.0):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.center_x = camera_width/2
        self.center_y = camera_height/2
        self.HFOV = HFOV
        self.VFOV = VFOV
        self.DFOV = DFOV
        
        self.tracker = Sort()
        cmap = plt.get_cmap('tab20b')
        # self.tracker_object = Tracker()
        self.colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]


    def process_frame(self, color_image, torch_model_object, detect_red):
        # conf_thres = 0.25  # Confidence threshold
        # Get bounding boxes
        results = torch_model_object(color_image)

        # Post process bounding boxes
        #rows = results.pandas().xyxy[0].to_numpy()
        img_size = max(color_image.shape)
        pad_x = max(color_image.shape[0] - color_image.shape[1], 0) * (img_size / max(color_image.shape))
        pad_y = max(color_image.shape[1] - color_image.shape[0], 0) * (img_size / max(color_image.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x

        detections_rows = results.pandas().xyxy

        for i in range(len(detections_rows)):
            rows = detections_rows[i].to_numpy()
        
        # if rows is not None:
        #     tracked_objects = self.tracker.update(rows)

            # unique_labels = rows[:, -1].cpu().unique()
            # n_cls_preds = len(unique_labels)
            # for x1, y1, x2, y2, obj_id, cls in tracked_objects:
        # if rows is not None:
        #     tracked_objects = self.tracker.update(rows)

            
      


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
            tracked_objects = self.tracker.update(np.expand_dims(rows[i], axis=0))
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                # print(type(y1))
                # box_h = int(((float(y2) - float(y1)) / unpad_h) * color_image.shape[0])
                # box_w = int(((float(x2) - float(x1)) / unpad_w) * color_image.shape[1])
                # y1 = int(((float(y1) - pad_y // 2) / unpad_h) * color_image.shape[0])
                # x1 = int(((float(x1) - pad_x // 2) / unpad_w) * color_image.shape[1])
                x1 = int(float(x1))
                y1 = int(float(y1))
                x2 = int(float(x2))
                y2 = int(float(y2))

                color = self.colors[int(obj_id) % len(self.colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 4)
                cv2.rectangle(color_image, (x1, y1-35), (x1+len(cls_pred)*19+60, y1), color, -1)
                cv2.putText(color_image, cls_pred + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
            # tracked_object = self.tracker.update(rows)
            # for x1, y1, x2, y2, obj_id, cls_pred in tracked_object:
            #     box_h = int(((y2 - y1) / unpad_h) * color_image.shape[0])
            #     box_w = int(((x2 - x1) / unpad_w) * color_image.shape[1])
            #     y1 = int(((y1 - pad_y // 2) / unpad_h) * color_image.shape[0])
            #     x1 = int(((x1 - pad_x // 2) / unpad_w) * color_image.shape[1])

            #     color = self.colors[int(obj_id) % len(self.colors)]
            #     color = [i * 255 for i in color]
            #     # cls = classes[int(cls_pred)]
            #     cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
            #     cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
            #     cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

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
                x_rad_offset, y_rad_offset = self.calculate_offset(((x_max+x_min)//2, (y_max+y_min)//2))
                color_image = self.write_bbx_frame(
                    color_image, bbox, label, conf)
                # tracked_objects = mot_tracker.update(detections.cpu())
                # kalman_prediction = self.tracker_object.track(bbox, hsv_roi, color_image)
                # color_image = self.write_bbx_frame(
                #     color_image, kalman_prediction, "kalman", conf="", color=(0,0,255), thickness=2
                # )
        # Display the image
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

    # default color green
    def write_bbx_frame(self, color_image, bbxs, label, conf="", color=(0, 255,0), thickness=2):
        # Display the bounding box
        x_min, y_min, x_max, y_max = bbxs
        cv2.rectangle(color_image, (x_min, y_min), 
            (x_max, y_max), color, thickness) 

        # Display the label with the confidence
        label_conf = label + " " + str(conf)
        cv2.putText(color_image, label_conf, (x_min,y_min),
             cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness, cv2.LINE_AA)

        return color_image

    def calculate_offset(self, roi_center_offsets):
        x_roi, y_roi = roi_center_offsets
        y_roi = self.camera_width - y_roi
        move_x = x_roi - self.center_x
        move_y = y_roi - self.center_y
        if(move_x != 0):
            move_x /= self.center_x
        if(move_y != 0):
            move_y /= self.center_y
        move_x *= self.HFOV/2
        move_y *= self.VFOV/2
        logger.debug("PROMPT2 x angle offset %f y angle offset %f", move_x, move_y)
        return (move_x, move_y)
