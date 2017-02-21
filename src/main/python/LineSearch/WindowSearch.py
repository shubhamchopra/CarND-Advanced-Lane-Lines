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

    def fullLineSearch(self, img):
        binary_img = np.zeros_like(img[:,:,0])
        binary_img[img[:,:,0] > 0] = 1
        histogram = np.sum(binary_img[binary_img.shape[0] / 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        window_height = np.int(binary_img.shape[0] / self.nwindows)
        nonzero = np.transpose(binary_img.nonzero())
        left_pixels = []
        right_pixels = []
        leftx_current = leftx_base
        rightx_current = rightx_base
        for w in range(self.nwindows):
            yhigh = binary_img.shape[0] - w*window_height
            ylow = binary_img.shape[0] - (w + 1)*window_height

            xleft_low = leftx_current - self.margin
            xleft_high = leftx_current + self.margin

            xright_low = rightx_current - self.margin
            xright_high = rightx_current + self.margin

            nonzero_left = nonzero[(nonzero[:,0] >= ylow) & (nonzero[:,0] < yhigh) & (nonzero[:,1] >= xleft_low) & (nonzero[:,1] < xleft_high)]
            nonzero_right = nonzero[(nonzero[:,0] >= ylow) & (nonzero[:,0] < yhigh) & (nonzero[:,1] >= xright_low) & (nonzero[:,1] < xright_high)]

            left_pixels.append(nonzero_left)
            right_pixels.append(nonzero_right)

            if len(nonzero_left) > self.minpix:
                leftx_current = np.int(np.mean(nonzero_left, axis = 0)[1])
            if len(nonzero_right) > self.minpix:
                rightx_current = np.int(np.mean(nonzero_right, axis = 0)[1])

        left_lane_pix = np.concatenate(left_pixels)
        right_lane_pix = np.concatenate(right_pixels)

        if self.sanityCheck(left_lane_pix[:, 0], left_lane_pix[:,1], right_lane_pix[:,0], right_lane_pix[:,1], img.shape) \
                or not (self.leftLane.initialized and self.rightLane.initialized):
            self.leftLane = self.leftLane.fitLine(left_lane_pix[:, 1], left_lane_pix[:, 0], fullLineSearch=True)
            self.rightLane = self.rightLane.fitLine(right_lane_pix[:, 1], right_lane_pix[:, 0], fullLineSearch=True)
        else:
            logging.info("Sanity check failure in full search. Using best fit")

    def getLaneIds(self, line, nonzero):
        xs = line.applyCurrent(nonzero[:, 0])
        ids = ((nonzero[:, 1] >= (xs - self.margin)) &
               (nonzero[:, 1] < (xs + self.margin)))
        return nonzero[ids]

    def lineSearch(self, img):
        binary_img = np.zeros_like(img[:, :, 0])
        binary_img[img[:, :, 0] > 0] = 1

        nonzero = np.transpose(binary_img.nonzero())

        leftLanePoints = self.getLaneIds(self.leftLane, nonzero)
        rightLanePoints = self.getLaneIds(self.rightLane, nonzero)

        if self.sanityCheck(leftLanePoints[:, 0], leftLanePoints[:, 1], rightLanePoints[:, 0], rightLanePoints[:, 1], img.shape):
            self.leftLane = self.leftLane.fitLine(leftLanePoints[:, 1], leftLanePoints[:, 0])
            self.rightLane = self.rightLane.fitLine(rightLanePoints[:, 1], rightLanePoints[:, 0])
        else:
            #need to do a full search
            logging.info("Doing a full search")
            self.fullLineSearch(img)


    def getLaneLines(self, img):
        if self.leftLane.initialized and self.rightLane.initialized:
            # logging.info("Using line information in lane search")
            self.lineSearch(img)
        else:
            # logging.info("Using full line search")
            self.fullLineSearch(img)

        laneStats = self.getLaneStats(img.shape[0], img.shape[1])
        return (laneStats, self.leftLane.current_fit, self.rightLane.current_fit)

    def showLanes(self, img, line):
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

    def showSearchResult(self, img):
        binary_img = np.zeros_like(img[:, :, 0])
        binary_img[img[:, :, 0] > 0] = 1

        nonzero = np.transpose(binary_img.nonzero())
        out_img = np.copy(img)
        leftLaneIds = self.getLaneIds(self.leftLane, nonzero)
        out_img[leftLaneIds[:,0], leftLaneIds[:,1]] = [255,0,0]
        rightLaneIds = self.getLaneIds(self.rightLane, nonzero)
        out_img[rightLaneIds[:,0], rightLaneIds[:,1]] = [0,0,255]

        window_img = np.zeros_like(img)
        self.showLanes(window_img, self.leftLane)
        self.showLanes(window_img, self.rightLane)

        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        (leftR, rightR, laneWidth, delta) = self.getLaneStats(img.shape[0], img.shape[1])
        str1 = "Left radius: {:.2f}m, right radius: {:.2f}m".format(leftR, rightR)
        logging.info(str1)
        str2 = "DeviationFromCenter: {:.2f}m Lane width: {:.2f}m".format(delta, laneWidth)
        logging.info(str2)

        cv2.putText(result, str1, (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
        cv2.putText(result, str2, (100, 250), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
        return result

    def getLaneStats(self, ymax, xmax):
        return HistogramSearch.getInitialLaneStats(self.leftLane.roc_fit, self.rightLane.roc_fit, ymax, xmax)

    def getInitialLaneStats(leftROCFit, rightROCFit, ymax, xmax):
        leftR = Line.getROC(leftROCFit, ymax)
        rightR = Line.getROC(rightROCFit, ymax)
        leftX = leftROCFit(ymax * Line.ym_per_pix)
        rightX = rightROCFit(ymax * Line.ym_per_pix)
        laneWidth = (rightX - leftX)
        delta = np.absolute(xmax * Line.xm_per_pix/ 2.0 - (leftX + rightX) / 2.0)
        return (leftR, rightR, laneWidth, delta)

    def sanityCheck(self, lefty, leftx, righty, rightx, shape):
        result = True
        if len(lefty) < HistogramSearch.minPointsForValidFit or len(righty) < HistogramSearch.minPointsForValidFit:
            logging.info("Not enough points, left = {}, right = {}".format(len(lefty), len(righty)))
            result = False
        else:
            leftFit = np.poly1d(np.polyfit(lefty, leftx, 2))
            leftROCFit = np.poly1d(np.polyfit(lefty * Line.ym_per_pix, leftx * Line.xm_per_pix, 2))
            rightFit = np.poly1d(np.polyfit(righty, rightx, 2))
            rightROCFit = np.poly1d(np.polyfit(righty * Line.ym_per_pix, rightx * Line.xm_per_pix, 2))
            (leftR, rightR, laneWidth, delta) = HistogramSearch.getInitialLaneStats(leftROCFit, rightROCFit, shape[0],
                                                                                    shape[1])
            if (laneWidth > 4 or laneWidth < 3) or (leftR < 500 or rightR < 500):
                logging.info("lane width: {}, leftR: {}, rightR: {}".format(laneWidth, leftR, rightR))
                result = False

        return result

if __name__ == "__main__":
    testImage = "output_images/perspective_transform/test_images/*.jpg"
    images = glob.glob(testImage)
    outputdir = "output_images/window_search/"

    for i, fname in enumerate(images):
        logging.info("Working on {}".format(fname))
        search = HistogramSearch()
        img = cv2.imread(fname)
        search.fullLineSearch(img)
        output = search.showSearchResult(img)
        outputFile = outputdir + fname
        # logging.info("Saving {}".format(outputFile))
        # cv2.imwrite(outputFile, output)
