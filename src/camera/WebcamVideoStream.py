#!/usr/bin/env python
# https://gist.github.com/allskyee/7749b9318e914ca45eb0a1000a81bf56
from threading import Thread, Lock
import cv2
import logging
import numpy as np
from pathlib import Path
import copy
logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.DEBUG)

class WebcamVideoStream :
    def __init__(self, src = 0) :
        self.stream = cv2.VideoCapture(src)
        # self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            logger.error("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = copy.copy(self.frame)
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def isOpened(self):
        ifOpen = True
        self.read_lock.acquire()
        if np.nansum(self.frame) == 0:
            ifOpen = False
        self.read_lock.release()
        return ifOpen
    
    def get(self, index=5):
        self.read_lock.acquire()
        attribute = copy.copy(self.stream.get(index))
        self.read_lock.release()
        return attribute

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()
        cv2.destroyAllWindows()

# if __name__ == "__main__" :
#     vs = WebcamVideoStream().start()
#     while True :
#         frame = vs.read()
#         cv2.imshow('webcam', frame)
#         if cv2.waitKey(1) == 27 :
#             break
#         print(f"fps = {vs.stream.get(5)}")

#     vs.stop()
#     cv2.destroyAllWindows()