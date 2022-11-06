# ignore this class
import cv2
import numpy as np
class Tracker:
    def __init__(self,):
        self.tracker = cv2.legacy.TrackerMedianFlow_create()
        self.seen_first = False