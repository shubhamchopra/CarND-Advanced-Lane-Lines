import numpy as np

class Line():
    '''
    Data structure to keep a track of previously estimated lines and update the best fit
    '''
    degree = 2
    ym_per_pix = 30.0 / 720
    xm_per_pix = 3.7 / 700

    def __init__(self):
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = None
        #radius of curvature fit
        self.roc_fit = None
        #radius of curvature fit
        self.roc_best_fit = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #iterations since full line search
        self.lastFullLineSearch = 0
        #check if a line was every initialized
        self.initialized = False

    def fitLine(self, xs, ys, fullLineSearch = False):
        '''
        We fit a line and update the best fit with the line
        We do a weighted average of the line co-efficients between the previous weighted fit
        and the current fit. This should have an exponential decay effect where the older fits
        fade away from the average.
        :param xs: Xs for polynomial fit
        :param ys: Ys for polynomial fit
        :param fullLineSearch: Was the xs and ys a result of full search?
        :return: self
        '''
        self.current_fit = Line.getFit(xs, ys)
        self.roc_fit = Line.getFit(xs*self.xm_per_pix, ys*self.ym_per_pix)
        if self.best_fit is None or self.roc_best_fit is None:
            self.best_fit = self.current_fit
            self.roc_best_fit = self.roc_fit
            self.initialized = True
        else:
            # this ensures we give higher weightage to the current fit and very old fits slowly
            # fade away from the best fit
            self.best_fit = (0.7*self.best_fit + 0.3*self.current_fit)
            self.roc_best_fit = (0.7*self.roc_best_fit + 0.3*self.roc_fit)

        if fullLineSearch:
            self.lastFullLineSearch = 0
        else:
            self.lastFullLineSearch += 1

        self.current_fit = self.best_fit
        self.roc_fit = self.roc_best_fit

        return self

    def applyCurrent(self, ys):
        return self.current_fit(ys)

    def getFit(xs, ys):
        '''
        static method to do a polynomial fit given xs and ys
        '''
        return np.poly1d(np.polyfit(ys, xs, Line.degree))

    def getROC(fit, y):
        '''
        Static method to return the radius of curvature for a given fit and a y value
        :param y:
        :return:
        '''
        first_d = fit.deriv(1)(y * Line.ym_per_pix)
        second_d = fit.deriv(2)(y * Line.ym_per_pix)
        r = (1 + first_d ** 2) ** 1.5 / np.abs(second_d)
        return r

    def getCurrentRadiusOfCurvature(self, y):
        r = Line.getROC(self.roc_fit, y)
        self.radius_of_curvature = r
        return r
