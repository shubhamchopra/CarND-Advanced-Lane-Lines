import numpy as np

class Line():
    degree = 2
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = None
        #radius of curvature fit
        self.roc_fit = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        #iterations since full line search
        self.lastFullLineSearch = 0

        self.ym_per_pix = 30/720
        self.xm_per_pix = 3.7/700

    def fitLine(self, xs, ys, fullLineSearch = False):
        self.current_fit = np.poly1d(np.polyfit(ys, xs, self.degree))
        self.roc_fit = np.poly1d(np.polyfit(ys*self.ym_per_pix, xs*self.xm_per_pix, self.degree))
        if self.best_fit is None:
            self.best_fit = self.current_fit
        else:
            # this ensures we give higher weightage to the current fit and very old fits slowly
            # fade away from the best fit
            self.best_fit = (self.best_fit + self.current_fit) / 2.0

        if fullLineSearch:
            self.lastFullLineSearch = 0
        else:
            self.lastFullLineSearch += 1

        return self

    def applyCurrent(self, ys):
        return self.current_fit(ys)

    def getCurrentRadiusOfCurvature(self, y):
        first_d = self.roc_fit.deriv(1)(y * self.ym_per_pix)
        second_d = self.roc_fit.deriv(2)(y * self.ym_per_pix)
        r = np.sqrt((1 + first_d**2)**3) / np.abs(second_d)
        self.radius_of_curvature = r
        return r
