import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
from LineSearch.Line import *

class HistogramSearch:
    def __init__(self):
        self.nwindows = 9
        self.margin = 100
        self.minpix = 100
        self.leftLane = Line()
        self.rightLane = Line()

    def fullLineSearch(self, img):
        # out_img = np.copy(img)
        binary_img = np.zeros_like(img[:,:,0])
        binary_img[img[:,:,0] > 0] = 1
        # plt.subplot(2, 1, 1)
        # plt.imshow(binary_img, cmap="gray")
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

            # cv2.rectangle(out_img, (xleft_low, ylow), (xleft_high, yhigh), (0,255,0), 2)
            # cv2.rectangle(out_img, (xright_low, ylow), (xright_high, yhigh), (0,255,0), 2)

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

        # self.leftLane = np.poly1d(np.polyfit(left_lane_pix[:,0], left_lane_pix[:,1], 2))
        # logging.info("Left fit polynomial : \n{}".format(self.leftLane))
        # self.rightLane = np.poly1d(np.polyfit(right_lane_pix[:,0], right_lane_pix[:,1], 2))
        # logging.info("Right fit polynomial : \n{}".format(self.rightLane))
        self.leftLane = self.leftLane.fitLine(left_lane_pix[:,1], left_lane_pix[:,0], fullLineSearch=True)
        self.rightLane = self.rightLane.fitLine(right_lane_pix[:,1], right_lane_pix[:,0], fullLineSearch=True)

        # ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
        # left_fitx = left_fit(ploty)
        # right_fitx = right_fit(ploty)
        # avg_fit = (0.5*left_fit + 0.5*right_fit)(ploty)
        # out_img[left_lane_pix[:,0], left_lane_pix[:,1]] = [255,0,0]
        # out_img[right_lane_pix[:,0], right_lane_pix[:,1]] = [0,0,255]
        # plt.subplot(2,1,2)
        # plt.imshow(out_img)
        # plt.plot(left_fitx, ploty, color = 'yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.plot(avg_fit, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720,0)
        # plt.show()

    def getLaneIds(self, line, nonzero):
        xs = line.applyCurrent(nonzero[:, 0])
        ids = ((nonzero[:, 1] >= (xs - self.margin)) &
               (nonzero[:, 1] < (xs + self.margin)))
        return nonzero[ids]

    def lineSearch(self, img):
        binary_img = np.zeros_like(img[:, :, 0])
        binary_img[img[:, :, 0] > 0] = 1

        nonzero = np.transpose(binary_img.nonzero())

        def getLaneFit(line, nonzero):
            lane_pix = self.getLaneIds(line, nonzero)
            return line.fitLine(lane_pix[:, 1], lane_pix[:, 0])

        self.leftLane = getLaneFit(self.leftLane, nonzero)
        self.rightLane = getLaneFit(self.rightLane, nonzero)

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
        leftR = self.leftLane.getCurrentRadiusOfCurvature(img.shape[0])
        rightR = self.rightLane.getCurrentRadiusOfCurvature(img.shape[0])
        leftX = self.leftLane.applyCurrent(img.shape[0])
        rightX = self.rightLane.applyCurrent(img.shape[0])
        delta = np.absolute(img.shape[1]/2.0 - (leftX + rightX)/2.0) * self.leftLane.xm_per_pix
        logging.info("Left radius: {}m, right radius: {}m, deviationFromCenter: {}m".format(leftR, rightR, delta))
        plt.imshow(result)
        plt.show()


if __name__ == "__main__":
    testImage = "output_images/perspective_transform/test_images/test3.jpg"
    img = cv2.imread(testImage)
    search = HistogramSearch()
    search.fullLineSearch(img)
    search.lineSearch(img)
    search.showSearchResult(img)
