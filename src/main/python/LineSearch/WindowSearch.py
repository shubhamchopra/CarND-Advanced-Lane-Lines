import numpy as np
import cv2
import glob
import logging
from LineSearch.Line import *

class HistogramSearch:
    minPointsForValidFit = 1000

    def __init__(self):
        self.nwindows = 9
        self.margin = 100
        self.minpix = 100
        self.leftLane = Line()
        self.rightLane = Line()

    def _fullLineSearch(self, img):
        '''
        We search the image by forming n windows in both lanes. We first find the centers of left and right lanes
        using histogram, and from there we build windows and try to locate pixels belonging to the lane markers
        :param img:
        :return:
        '''
        # we get the binary image
        binary_img = np.zeros_like(img[:,:,0])
        binary_img[img[:,:,0] > 0] = 1
        # using histogram, we find left and right regions where we can start searching for lane lines
        histogram = np.sum(binary_img[binary_img.shape[0] / 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] / 2)
        # we get the left base and right base to begin searching
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = np.int(binary_img.shape[0] / self.nwindows)
        nonzero = np.transpose(binary_img.nonzero())
        left_pixels = []
        right_pixels = []
        leftx_current = leftx_base
        rightx_current = rightx_base
        for w in range(self.nwindows):
            # for each window, we get the boundaries
            yhigh = binary_img.shape[0] - w*window_height
            ylow = binary_img.shape[0] - (w + 1)*window_height

            xleft_low = leftx_current - self.margin
            xleft_high = leftx_current + self.margin

            xright_low = rightx_current - self.margin
            xright_high = rightx_current + self.margin

            # we find out non-zero elements in the binary image within the window boundaries
            nonzero_left = nonzero[(nonzero[:,0] >= ylow) & (nonzero[:,0] < yhigh) & (nonzero[:,1] >= xleft_low) &
                                   (nonzero[:,1] < xleft_high)]
            nonzero_right = nonzero[(nonzero[:,0] >= ylow) & (nonzero[:,0] < yhigh) & (nonzero[:,1] >= xright_low) &
                                    (nonzero[:,1] < xright_high)]

            #we keep a track of all the points we found
            left_pixels.append(nonzero_left)
            right_pixels.append(nonzero_right)

            # if we were able to find a good number of points in either lanes, we update the center of the next
            # search window
            if len(nonzero_left) > self.minpix:
                leftx_current = np.int(np.mean(nonzero_left, axis = 0)[1])
            if len(nonzero_right) > self.minpix:
                rightx_current = np.int(np.mean(nonzero_right, axis = 0)[1])

        left_lane_pix = np.concatenate(left_pixels)
        right_lane_pix = np.concatenate(right_pixels)

        # we check if the points we found make sense
        if self._sanityCheck(left_lane_pix[:, 0], left_lane_pix[:, 1], right_lane_pix[:, 0], right_lane_pix[:, 1], img.shape) \
                or not (self.leftLane.initialized and self.rightLane.initialized):
            # the points seem ok, so we update the line fits
            self.leftLane = self.leftLane.fitLine(left_lane_pix[:, 1], left_lane_pix[:, 0], fullLineSearch=True)
            self.rightLane = self.rightLane.fitLine(right_lane_pix[:, 1], right_lane_pix[:, 0], fullLineSearch=True)
        else:
            # the points were bad, so we don't pollute our line fits with the new data, and use the
            # previous best fit
            logging.info("Sanity check failure in full search. Using best fit")

    def _getLaneIds(self, line, nonzero):
        xs = line.applyCurrent(nonzero[:, 0])
        ids = ((nonzero[:, 1] >= (xs - self.margin)) &
               (nonzero[:, 1] < (xs + self.margin)))
        return nonzero[ids]

    def _lineSearch(self, img):
        '''
        This function just searches for lanes around the previous fit. This would be a lot faster than doing a full
        window search. We do a sanity check to confirm the new fits make sense, else we fall back on full window
        search
        :param img:
        :return:
        '''
        binary_img = np.zeros_like(img[:, :, 0])
        binary_img[img[:, :, 0] > 0] = 1

        nonzero = np.transpose(binary_img.nonzero())

        leftLanePoints = self._getLaneIds(self.leftLane, nonzero)
        rightLanePoints = self._getLaneIds(self.rightLane, nonzero)

        if self._sanityCheck(leftLanePoints[:, 0], leftLanePoints[:, 1], rightLanePoints[:, 0], rightLanePoints[:, 1], img.shape):
            self.leftLane = self.leftLane.fitLine(leftLanePoints[:, 1], leftLanePoints[:, 0])
            self.rightLane = self.rightLane.fitLine(rightLanePoints[:, 1], rightLanePoints[:, 0])
        else:
            #need to do a full search
            logging.info("Doing a full search")
            self._fullLineSearch(img)


    def getLaneLines(self, img):
        '''
        The entry point of processing. We take in a thresholded, perspective transformed image as input.
        If this was the first image, we do a full search, else we do a line search where we only search
        around a previous fit
        :param img:
        :return: lane stats, left lane fit, right lane fit
        '''
        if self.leftLane.initialized and self.rightLane.initialized:
            self._lineSearch(img)
        else:

            self._fullLineSearch(img)

        laneStats = self._getLaneStats(img.shape[0], img.shape[1])
        return (laneStats, self.leftLane.current_fit, self.rightLane.current_fit)

    def _getLaneStats(self, ymax, xmax):
        return HistogramSearch._getInitialLaneStats(self.leftLane.roc_fit, self.rightLane.roc_fit, ymax, xmax)

    def _getInitialLaneStats(leftROCFit, rightROCFit, ymax, xmax):
        leftR = Line.getROC(leftROCFit, ymax)
        rightR = Line.getROC(rightROCFit, ymax)
        leftX = leftROCFit(ymax * Line.ym_per_pix)
        rightX = rightROCFit(ymax * Line.ym_per_pix)
        laneWidth = (rightX - leftX)
        delta = np.absolute(xmax * Line.xm_per_pix/ 2.0 - (leftX + rightX) / 2.0)
        return (leftR, rightR, laneWidth, delta)

    def _sanityCheck(self, lefty, leftx, righty, rightx, shape):
        '''
        We check for fit quality here.
        '''
        result = True
        if len(lefty) < HistogramSearch.minPointsForValidFit or len(righty) < HistogramSearch.minPointsForValidFit:
            logging.info("Not enough points, left = {}, right = {}".format(len(lefty), len(righty)))
            result = False
        else:
            leftROCFit = Line.getFit(leftx * Line.xm_per_pix, lefty * Line.ym_per_pix)
            rightROCFit = Line.getFit(rightx * Line.xm_per_pix, righty * Line.ym_per_pix)
            (leftR, rightR, laneWidth, delta) = HistogramSearch._getInitialLaneStats(leftROCFit, rightROCFit, shape[0],
                                                                                     shape[1])
            if (laneWidth > 4 or laneWidth < 3.3) or (leftR < 500 or rightR < 500):
                # if the lane width is too wide or too narrow, or the radius of curvature is too low on either lanes
                # we conclude that the fit has a problem
                logging.info("lane width: {}, leftR: {}, rightR: {}".format(laneWidth, leftR, rightR))
                result = False

        return result

    def _showLanes(self, img, line):
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        xs = line.applyCurrent(ploty)
        line_window1 = np.array([np.transpose(np.vstack([xs - self.margin, ploty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([xs + self.margin, ploty])))])
        line_pts = np.hstack((line_window1, line_window2))
        # img[np.int32(ploty), np.int32(xs)] = [255, 255, 255]

        cv2.fillPoly(img, np.int_([line_pts]), (0, 255, 0))
        line_window1 = np.array([np.transpose(np.vstack([xs - 5, ploty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([xs + 5, ploty])))])
        line_pts = np.hstack((line_window1, line_window2))
        cv2.fillPoly(img, np.int_([line_pts]), (0,0,0))

    def _showSearchResult(self, img):
        binary_img = np.zeros_like(img[:, :, 0])
        binary_img[img[:, :, 0] > 0] = 1

        nonzero = np.transpose(binary_img.nonzero())
        out_img = np.copy(img)
        leftLaneIds = self._getLaneIds(self.leftLane, nonzero)
        out_img[leftLaneIds[:,0], leftLaneIds[:,1]] = [255,0,0]
        rightLaneIds = self._getLaneIds(self.rightLane, nonzero)
        out_img[rightLaneIds[:,0], rightLaneIds[:,1]] = [0,0,255]

        window_img = np.zeros_like(img)
        self._showLanes(window_img, self.leftLane)
        self._showLanes(window_img, self.rightLane)

        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        (leftR, rightR, laneWidth, delta) = self._getLaneStats(img.shape[0], img.shape[1])
        str1 = "Left radius: {:.2f}m, right radius: {:.2f}m".format(leftR, rightR)
        logging.info(str1)
        str2 = "DeviationFromCenter: {:.2f}m Lane width: {:.2f}m".format(delta, laneWidth)
        logging.info(str2)

        cv2.putText(result, str1, (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
        cv2.putText(result, str2, (100, 250), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
        return result

if __name__ == "__main__":
    testImage = "output_images/perspective_transform/test_images/*.jpg"
    images = glob.glob(testImage)
    outputdir = "output_images/window_search/"

    for i, fname in enumerate(images):
        logging.info("Working on {}".format(fname))
        search = HistogramSearch()
        img = cv2.imread(fname)
        search._fullLineSearch(img)
        output = search._showSearchResult(img)
        outputFile = outputdir + fname
        # logging.info("Saving {}".format(outputFile))
        # cv2.imwrite(outputFile, output)
