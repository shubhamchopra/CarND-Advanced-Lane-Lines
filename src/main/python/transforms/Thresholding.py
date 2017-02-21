from transforms.Transform import Transform
import numpy as np
import cv2
import glob

import logging


class Thresholding(Transform):
    def __init__(self, kernel_size = 3, sobel_thresh = (0, 255), s_thresh = (0, 255)):
        self._sobel_thresh = sobel_thresh
        self._s_thresh = s_thresh
        self._kernel_size = kernel_size

    def _transformFunc(self, image):
        def sobelX(im):
            return np.absolute(cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=self._kernel_size))
        def sobelY(im):
            return np.absolute(cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=self._kernel_size))
        def sobel(dim, im):
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            if dim == 'x':
                return sobelX(gray)
            elif dim == 'y':
                return sobelY(gray)
            else:
                X = sobelX(gray)
                Y = sobelY(gray)
                return np.sqrt(np.square(X) + np.square(Y))
        def sChannel(im):
            hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
            return hls[:, :, 2]
        def scaleAndThreshold(im, threshold):
            image = np.uint8(im * 255.0 / np.max(im))
            mask = np.zeros_like(image)
            mask[(image >= threshold[0]) & (image <= threshold[1])] = 1
            return mask

        sobelIm = sobel("x", image)
        sobelMask = scaleAndThreshold(sobelIm, self._sobel_thresh)

        sChan = sChannel(image)
        sMask = scaleAndThreshold(sChan, self._s_thresh)

        combined = np.zeros_like(sChan)
        combined[(sMask == 1) | (sobelMask == 1)] = 255
        return np.dstack((combined, combined, combined))

    def transform(self, img):
        dst = self._transformFunc(img)
        return dst


if __name__ == "__main__":
    imgNames = glob.glob("test_images/*.jpg")
    transform = Thresholding(kernel_size=5, sobel_thresh=(40, 100), s_thresh=(150, 255))
    for i, imName in enumerate(imgNames):
        img = cv2.imread(imName)
        dst = transform.transform(img)
        dstFile = "output_images/thresholding/" + imName
        logging.info("Writing file {}".format(dstFile))
        cv2.imwrite(dstFile, dst)