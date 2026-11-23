import numpy as np

class Line():
    def __init__(self):
        #上一轮迭代中是否检测该行
        self.detected = False
        #直线n次拟合的x的值
        self.recent_fitted = []
        #拟合线在过去 n 次迭代中的 x 平均值
        self.bestx = None
        #过去n次迭代中平均的多项式系数
        self.best_fit = None
        #最近拟合的多项次系数
        self.current_fit = [np.array([False])]
        #以某种单位表示的直线曲率半径
        self.radius_of_curvature = None
        #车辆中心到该线的距离（单位：米）
        self.line_base_pos = None
        #上次拟合和新拟合之间的拟合系数差异
        self.diffs = np.array([0,0,0],dtype='float')
        #检测到的直线像素的x值
        self.allx = None
        #检测到的直线像素dey值
        self.ally = None
        pass

    def check_detected(self):
        if (self.diffs[0]<0.01 and self.diffs[1]<10.0 and self.diffs[2]<1000.0) and (len(self.recent_fitted)>0):
            return True
        else:
            return False


    def update(self,fit):
        if fit is not None:
            if self.best_fit is not None:
                self.diffs = abs(fit - self.best_fit)
                if self.check_detected():
                    self.detected = True
                    if len(self.recent_fitted)>10:
                        self.recent_fitted = self.recent_fitted[1:]
                        self.recent_fitted.append(fit)
                    else:
                        self.recent_fitted.append(fit)
                    self.best_fit = np.average(self.recent_fitted,axis=0)
                    self.current_fit = fit
                else:
                    self.detected = False
            else:
                self.best_fit = fit
                self.current_fit = fit
                self.detected = True
                self.recent_fitted.append(fit)

