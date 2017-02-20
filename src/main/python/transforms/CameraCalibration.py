from transforms.Transform import Transform
import numpy as np
import cv2
import glob

import logging


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

if __name__ == "__main__":
    images = "camera_cal/*.jpg"

    imgNames = glob.glob(images)
    cameraCal = CameraCalibration(images, (9, 6))

    for i, imName in enumerate(imgNames):
        img = cv2.imread(imName)
        logging.info("Image {} size {}".format(imName, img.shape))
        dst = cameraCal.transform(img)
        outFilename = "output_images/" + imName
        logging.info("Writing file {}".format(outFilename))
        cv2.imwrite(outFilename, dst)