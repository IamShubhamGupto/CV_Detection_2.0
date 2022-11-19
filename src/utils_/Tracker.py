# https://github.com/tobybreckon/python-examples-cv/blob/master/kalman_tracking_live.py
import cv2
import numpy as np

# return centre of a set of points representing a rectangle
def center(points):
    x = np.float32(
        (points[0][0] +
         points[1][0] +
         points[2][0] +
         points[3][0]) /
        4.0)
    y = np.float32(
        (points[0][1] +
         points[1][1] +
         points[2][1] +
         points[3][1]) /
        4.0)
    return np.array([np.float32(x), np.float32(y)], np.float32)
class Tracker(object):
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0]], np.float32)

        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)

        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32) * 0.03
        self.prediction = np.zeros((2, 1), np.float32)
        self.s_lower = 60
        # cv2.createTrackbar("s lower", window_name2, s_lower, 255, nothing)
        self.s_upper = 255
        # cv2.createTrackbar("s upper", window_name2, s_upper, 255, nothing)
        self.v_lower = 32
        # cv2.createTrackbar("v lower", window_name2, v_lower, 255, nothing)
        self.v_upper = 255
        # cv2.createTrackbar("v upper", window_name2, v_upper, 255, nothing)
        # Setup the termination criteria for search, either 10 iteration or
        # move by at least 1 pixel pos. difference
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        # self.hsv_crop = hsv_crop
    
    def track(self, boxes, hsv_crop, frame):
        # initial tracking
        track_window = (
                    boxes[0],
                    boxes[1],
                    boxes[2] - boxes[0],
                    boxes[3] - boxes[1]
                    )
        x, y, w, h = track_window
        # select all Hue (0-> 180) and Sat. values but eliminate values
        # with very low saturation or value (due to lack of useful
        # colour information)
        mask = cv2.inRange(
            hsv_crop, np.array(
                (0., float(self.s_lower), float(self.v_lower))), np.array(
                (180., float(self.s_upper), float(self.v_upper))))

        # construct a histogram of hue and saturation values and
        # normalize it
        crop_hist = cv2.calcHist(
            [hsv_crop], [
                0, 1], mask, [
                180, 255], [
                0, 180, 0, 255])
        cv2.normalize(crop_hist, crop_hist, 0, 255, cv2.NORM_MINMAX)

        # convert incoming image to HSV
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # back projection of histogram based on Hue and Saturation only
        img_bproject = cv2.calcBackProject(
            [img_hsv], [
                0, 1], crop_hist, [
                0, 180, 0, 255], 1)
        ret, track_window = cv2.CamShift(
                img_bproject, track_window, self.term_crit)
        
        # extract centre of this observation as points
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)

        # use to correct kalman filter
        self.kalman.correct(center(pts))

        # get new kalman filter prediction
        self.prediction = self.kalman.predict()

        return np.array([int(self.prediction[0] - (0.5 * w)), int(self.prediction[1] - (0.5 * h)), int(self.prediction[0] + (0.5 * w)), int(self.prediction[1] + (0.5 * h))])
        # frame = cv2.rectangle(frame,
        #                           (int(prediction[0] - (0.5 * w)), int(prediction[1] - (0.5 * h))),
        #                           (int(prediction[0] + (0.5 * w)), int(prediction[0] + (0.5 * w))),
        #                           (0, 255, 0), 2)
        # return frame

        