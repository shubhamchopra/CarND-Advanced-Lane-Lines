import numpy as np
import cv2
import glob

import logging

class Transform:
    def __init__(self):
        pass

    def transform(self, img):
        return img

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

        sobelIm = sobel("x", image)
        sobelIm = np.uint8(sobelIm * 255.0 / np.max(sobelIm))
        sobelMask = np.zeros_like(sobelIm)
        sobelMask[(sobelIm >= self._sobel_thresh[0]) & (sobelIm <= self._sobel_thresh[1])] = 1

        sChan = sChannel(image)
        sMask = np.zeros_like(sChan)

        sMask[(sChan >= self._s_thresh[0]) & (sChan <= self._s_thresh[1])] = 1

        combined = np.zeros_like(sChan)
        combined[(sMask == 1) | (sobelMask == 1)] = 255
        return combined

    def transform(self, img):
        dst = self._transformFunc(img)
        return dst


class CameraCalibration(Transform):

    def __init__(self, images_glob, chessBoardShape):
        '''
        Create camera calibration transformation
        :param images_glob: Images of chessboards with appropriate sizes in glob format.
                            For example: data/CB*.jpg
        :param chessBoardShape: Size of the chessboard in the images, expressed as a tuple
        '''
        self._images = glob.glob(images_glob)
        self._chessBoardShape = chessBoardShape
        self._imgSize = None
        self._calibrateCamera()


    def _calibrateCamera(self):
        cbX = self._chessBoardShape[0]
        cbY = self._chessBoardShape[1]
        objp = np.zeros((cbX * cbY, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cbX, 0:cbY].T.reshape(-1, 2)
        objPoints = []
        imgPoints = []

        for i, fname in enumerate(self._images):
            corners = self._generateImgPoints(fname)
            if len(corners) > 0:
                objPoints.append(objp)
                imgPoints.append(corners)

        if len(imgPoints) > 0:
            # we only calibrate if we were able to detect image points
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, self._imgSize, None, None)
            self._cameraCalibrationData = {"ret": ret, "mtx": mtx, "dist": dist, "rvecs": rvecs, "tvecs": tvecs}
            logging.info("Calibrating for image size {}".format(self._imgSize))
        else:
            raise RuntimeError("Unable to detect chess board points in these images {}".format(self._images))

        if not self._cameraCalibrationData["ret"]:
            raise RuntimeError("Unable to calibrate with the images in {}".format(self._images))


    def _generateImgPoints(self, fname):
        img = cv2.imread(fname)
        if self._imgSize is None:
            self._imgSize = img.shape[0:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        cbX = self._chessBoardShape[0]
        cbY = self._chessBoardShape[1]
        ret, corners = cv2.findChessboardCorners(gray, (cbX, cbY), None)

        # If found, add object points, image points
        if ret == True:
            logging.info("Found corners for image {}".format(fname))
            return corners
        else:
            logging.warning("No corners found for image {}".format(fname))
            return []


    def transform(self, img):
        imgSize = img.shape[0:2]
        srcImage = img
        if imgSize != self._imgSize:
            logging.warning("Camera calibrated for images of size {} but being used on an image of size {}. Resizing image".format(self._imgSize, imgSize))
            srcImage = cv2.resize(img, (self._imgSize[1], self._imgSize[0]))

        mtx = self._cameraCalibrationData["mtx"]
        dist = self._cameraCalibrationData["dist"]
        dst = cv2.undistort(srcImage, mtx, dist, None, mtx)
        return dst
