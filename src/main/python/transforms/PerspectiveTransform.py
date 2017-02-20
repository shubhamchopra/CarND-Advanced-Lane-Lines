from transforms.Transform import Transform
from transforms.CameraCalibration import CameraCalibration
from transforms.Thresholding import Thresholding
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import logging

class PerspectiveTransform(Transform):
    X = 1280
    Y = 720
    pts = np.float32([[200 + 20, Y], [X / 2 - 80 - 5, 2 * Y / 3], [X / 2 + 80 + 10, 2 * Y / 3], [X - 200 + 20, Y]])
    dstpts = np.float32([[300, Y], [300, 0], [X - 300, 0], [X - 300, Y]])

    def __init__(self, srcCoords = pts, dstCoords = dstpts):
        self._M = cv2.getPerspectiveTransform(srcCoords, dstCoords)
        self._Minv = cv2.getPerspectiveTransform(dstCoords, srcCoords)

    def transform(self, img):
        warped = cv2.warpPerspective(img, self._M, None, flags=cv2.INTER_LINEAR)
        return warped

    def invTransform(self, img):
        warped = cv2.warpPerspective(img, self._Minv, None, flags=cv2.INTER_LINEAR)
        return warped

if __name__ == "__main__":
    testImages = "test_images/*.jpg"
    outputDir = "output_images/perspective_transform/"
    cameraCalImages = "camera_cal/*.jpg"
    cameraCal = CameraCalibration(cameraCalImages, (9, 6))
    X = 1280
    Y = 720
    pts = np.float32([[200 + 20, Y], [X / 2 - 80 - 5, 2 * Y / 3], [X / 2 + 80 + 10, 2 * Y / 3], [X - 200 + 20, Y]])
    dstpts = np.float32([[300, Y], [300, 0], [X - 300, 0], [X - 300, Y]])
    perspTrans = PerspectiveTransform(pts, dstpts)
    threshTrans = Thresholding(kernel_size=5, sobel_thresh=(40, 100), s_thresh=(150, 255))
    for i, fname in enumerate(glob.glob(testImages)):
        # img = plt.imread(testImage)
        # plt.subplot(2, 2, 1)
        # plt.imshow(img)
        # plt.title("Original Image")
        img = plt.imread(fname)
        img = cameraCal.transform(img)
        img = threshTrans.transform(img)
        # plt.subplot(2, 2, 2)
        # plt.imshow(img)
        # plt.title("Undistorted")

        transformed = perspTrans.transform(img)
        # transformed = cv2.polylines(transformed, [np.array(dstpts, np.int32)], isClosed= True, color=(255, 0, 0), thickness=3)
        # plt.subplot(2, 2, 3)
        # dst = cv2.polylines(img, [np.array(pts, np.int32)], isClosed=True, color=(255, 0, 0), thickness=3)
        # plt.imshow(dst)
        # plt.title("With polygon lines")
        # plt.subplot(2, 2, 4)
        # plt.imshow(transformed)
        # plt.title("with perspective transform")
        # plt.show()
        dstName = outputDir + fname
        logging.info("Saving {}".format(dstName))
        plt.imsave(fname = dstName, arr = transformed, cmap = "gray")
