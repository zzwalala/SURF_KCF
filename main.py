import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
import cv2
from getrect import get_rect
from PyQt5 import QtGui
from tracking import Ui_MainWindow

import numpy as np
from kcf import Tracker


def find_target(target, frame):
    target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Initiate SURF detector
    surf = cv2.xfeatures2d.SURF_create(10)  # Hessian thershold

    # find the keypoints and descriptors with SIFT
    surf.setUpright(True)
    target_key_query, target_desc_query = surf.detectAndCompute(target, None)
    frame_key_query, frame_desc_query = surf.detectAndCompute(frame, None)

    #print(frame_desc_query)
    # 使用KDTree进行检索
    # FLANN参数
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    # 使用FlannBasedMatcher 寻找最近邻近似匹配
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # 使用knnMatch匹配处理，返回matches
    matches = flann.knnMatch(target_desc_query, frame_desc_query, k=2)

    # store all the good matches as per Lowe's ratio test.
    filtered = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            filtered.append(m)

    if len(filtered) > 2:  # number_threshold
        src_pts = np.float32(
            [target_key_query[m.queryIdx].pt for m in filtered]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [frame_key_query[m.trainIdx].pt for m in filtered]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = target.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        midpoints = []
        for i in range(dst.shape[0]):
            if i + 1 <= 3:
                j = i + 1
            else:
                j = 0
            midpoints.append((dst[i] + dst[j]) / 2)
        midpoints = np.array(midpoints).reshape(-1, 2)

        dst = np.array([[midpoints[0][0], midpoints[3][1]],
                        [midpoints[0][0], midpoints[1][1]],
                        [midpoints[2][0], midpoints[1][1]],
                        [midpoints[2][0], midpoints[3][1]]])

        dst = np.int32(dst)
        return dst[0][0], dst[0][1], dst[3][0] - dst[0][0], dst[1][1] - dst[0][1]
    else:
        print("Warning. No enough features between the input image and the video first frame to calculate.")



# self.init_x = init[0][0]
# self.init_y = init[0][1]
# self.width = init[3][0] - init[0][0]
# self.height = init[1][1] - init[0][1]

def cvImgtoQtImg(cvImg):  # 定义opencv图像转PyQt图像的函数
    QtImgBuf = cv2.cvtColor(cvImg, cv2.COLOR_BGR2BGRA)

    QtImg = QtGui.QImage(QtImgBuf.data, QtImgBuf.shape[1], QtImgBuf.shape[0], QtGui.QImage.Format_RGB32)

    return QtImg

class ObjectTrackingGUI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.timer = QTimer(self)
        self.setupUi(self)
        self.call_back_functions()
        self.cap = []
        self.timer_camera = QTimer()
        self.bClose = False
        self.read_image = None
        self.read_flag = False
        self.cut_flag = False
        self.init_x = 0
        self.init_y = 0
        self.width = 0
        self.height = 0

    def call_back_functions(self):
        self.pushButton.clicked.connect(self.open_image)
        self.pushButton_2.clicked.connect(self.open_video)
        self.pushButton_3.clicked.connect(self.start_func)
        self.pushButton_4.clicked.connect(self.close_window)
        self.pushButton_5.clicked.connect(self.get_image)

    def open_image(self):
        img_name, img_type = QFileDialog.getOpenFileName(
            self, "打开图片", "", "All Files(*)")
        jpg = QtGui.QPixmap(img_name).scaled(self.label.width(), self.label.height())
        self.read_image = cv2.imread(img_name)

        if self.read_image is None:
            print("Open a correct form of pictures, like jpg, png and so on.")
        print(1111)
        self.read_flag = True
        self.label.setPixmap(jpg)



    def open_video(self):
        videoName, _ = QFileDialog.getOpenFileName(self, "Open", "", "All Files(*);;*.avi;;*.mp4")
        self.cap = cv2.VideoCapture(videoName)
        if not self.cap.isOpened():
            print("Open a correct form of videos.")
            exit()

        flag, img_rd = cv2.VideoCapture(videoName).read()
        img_rd_1 = cvImgtoQtImg(img_rd)
        jpg_1 = QtGui.QPixmap(img_rd_1).scaled(self.label_2.width(), self.label_2.height())
        self.label_2.setPixmap(jpg_1)


    def start_func(self):
        if not ((self.cut_flag == True and self.read_flag == False) or (self.cut_flag == False and self.read_flag \
                                                                        == True)):
            print("Either choose a photo or cut the video frame first. You are not allowed to do both of these at the "
                  "same time.")
            pass
        else:
            ret, frame = self.cap.read()
            if self.read_flag == True:
                self.init_x,self.init_y, self.width, self.height = find_target(self.read_image, frame)
            tracker = Tracker()
            rectangle = [self.init_x, self.init_y, self.width, self.height]
            tracker.init(frame, rectangle)
            while not self.bClose:
                ret, frame = self.cap.read()  # 逐帧读取影片
                if not ret:
                    if frame is None:
                        print("The video has ended.")
                    else:
                        print("Read video error!")
                    break
                x, y, w, h = tracker.update(frame)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                QtImg = cvImgtoQtImg(frame)  # 将帧数据转换为PyQt图像格式
                show_Img = QtGui.QPixmap(QtImg).scaled(self.label_2.width(), self.label_2.height())
                self.label_2.setPixmap(show_Img)
                c = cv2.waitKey(1) & 0xFF
                if c == 27 or c == ord('q'):
                    break
            self.read_flag = False
            self.cut_flag = False

    def get_image(self):
        ret, frame = self.cap.read()  # 逐帧读取影片
        cut_image, self.init_x, self.init_y, self.width, self.height = get_rect(frame, title='get_rect')
        QtImg = cvImgtoQtImg(cut_image)  # 将帧数据转换为PyQt图像格式
        img = QtGui.QPixmap(QtImg).scaled(self.label.width(), self.label.height())
        self.cut_flag = True
        self.label.setPixmap(img)

    def close_window(self):
        self.cap.release()
        self.timer.stop()
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    tracking = ObjectTrackingGUI()
    tracking.show()
    sys.exit(app.exec_())