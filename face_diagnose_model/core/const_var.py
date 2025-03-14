import os

import numpy as np

BASE_PATH = os.path.abspath("../")
SKIN_PARAM_PATH = BASE_PATH + "/designer/param.json"
OUTPUT_PATH = BASE_PATH + "/result"
# OPENCV_CASCADE_PATH = "G:\\PyCharmProgram\\FaceDetection\\FaceDetect"
OPENCV_CASCADE_PATH = r"D:\software\PyCharm\envs\deepLearningpy37\Lib\site-packages\cv2\data"
FONT_PATH = BASE_PATH + "/fonts/simsun.ttc"
COLORDICT = {'blue': (255, 0, 0),
             'green': (0, 255, 0),
             'red': (0, 0, 255),
             'yellow': (0, 255, 255),
             'magenta': (255, 0, 255),
             'cyan': (255, 255, 0),
             'white': (255, 255, 255),
             'black': (0, 0, 0),
             'gray': (125, 125, 125),
             'rand': np.random.randint(0, high=256, size=(3,)).tolist(), 'dark_gray': (50, 50, 50),
             'light_gray': (220, 220, 220)}

KEY_ting = "ting"
KEY_que = "que"
KEY_quan_left = "quan_left"
KEY_quan_right = "quan_right"
KEY_ming_tang = "ming_tang"
KEY_jia_left = "jia_left"
KEY_jia_right = "jia_right"
KEY_chun = "chun"
KEY_ke = "ke"
CAMERA_NUMBER_PORT = 0

FACIAL_LANDMARKS_NAME_DICT = {
    KEY_ting: "庭",  # 额头上面
    KEY_que: "阙",  # 眉毛中间
    KEY_quan_left: "颧左",  # 眼角旁边
    KEY_quan_right: "颧右",  # 眼角旁边
    KEY_ming_tang: "明堂",  # 鼻头
    KEY_jia_left: "颊左",  # 脸颊左
    KEY_jia_right: "颊右",  # 脸颊右
    KEY_chun: "唇部",  # 脸颊右
    KEY_ke: "颏"  # 下巴到下唇中间
}

VIDEO_RGB = "RGB"
VIDEO_Lab = "Lab"
VIDEO_HSV = "HSV"
VIDEO_YCrCb = "YCrCb"
VIDEO_Melt = "Melt"

KEY_KernelSize = "KernelSize"
KEY_Iterations = "Iterations"
KEY_THRESHOLD_RANGE_MIN = "tmin"
KEY_THRESHOLD_RANGE_MAX = "tmax"

KEY_MIN_RANGE = "range_min"
KEY_MAX_RANGE = "range_max"