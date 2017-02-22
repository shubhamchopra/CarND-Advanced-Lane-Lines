from transforms.CameraCalibration import CameraCalibration
from transforms.PerspectiveTransform import PerspectiveTransform
from transforms.Thresholding import Thresholding
from LineSearch.WindowSearch import HistogramSearch
import logging
import cv2
from moviepy.editor import VideoFileClip
import numpy as np

logging.basicConfig(level=logging.INFO)

class Main:
    cameraCalImages = "camera_cal/*.jpg"
    cameraCal = CameraCalibration(cameraCalImages, (9, 6))
    perspTrans = PerspectiveTransform()
    threshTrans = Thresholding(kernel_size=5, sobel_thresh=(40, 100), s_thresh=(150, 255))
    laneSearch = HistogramSearch()

    def process_image(self, image):
        '''
        The main processing function that takes an image, applies all the transforms
        and returns an image with the overlay of green lane marker.
        :param image:
        :return: a transformed image with lane markings
        '''
        undist = self.cameraCal.transform(image)
        img = self.threshTrans.transform(undist)
        img = self.perspTrans.transform(img)
        laneStats, leftFit, rightFit = self.laneSearch.getLaneLines(img)
        leftR, rightR, laneWidth, delta = laneStats

        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        leftx = leftFit(ploty)
        rightx = rightFit(ploty)

        pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        dst = np.zeros_like(image)
        cv2.fillPoly(dst, np.int_([pts]), (0, 255, 0))

        invWarp = self.perspTrans.invTransform(dst)

        dst = cv2.addWeighted(undist, 1, invWarp, 0.3, 0)
        str1 = "Left ROC: {:.2f}m, Right ROC: {:.2f}m".format(leftR, rightR)
        str2 = "Deviation From Center: {:.2f}m, Lane width: {:.2f}m".format(delta, laneWidth)
        cv2.putText(dst, str1, (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
        cv2.putText(dst, str2, (100, 250), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
        return dst

    def process_video(self, inputVid, outputVid):
        clip1 = VideoFileClip(inputVid)
        laneClip = clip1.fl_image(self.process_image)
        laneClip.write_videofile(outputVid, audio=False)


if __name__ == "__main__":
    mainRunner = Main()
    input = "project_video.mp4"
    output = "project_video_output.mp4"
    mainRunner.process_video(input, output)
