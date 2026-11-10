import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def sortContours(rect):
    return sorted(rect,key=lambda c:cv.boundingRect(c)[0])
img = cv.imread('project3_1_temp.png')
ref = cv.cvtColor(img,cv.COLOR_BGRA2GRAY)

#转为二值图
ref = cv.threshold(ref,10,255,cv.THRESH_BINARY_INV)[1]

#只检测外轮廓
refCnts,hierarchy = cv.findContours(ref.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img,refCnts,-1,(0,0,255),3) #对模版的10个数字画出外轮廓
refCnts = sortContours(refCnts) #从左到右，从上到下

#每个数字对应一个模版
digits = {}
for (i,c) in enumerate(refCnts):
    '''
    计算外接矩形并且resize成合适大小
    '''
    (x,y,w,h) = cv.boundingRect(c) #外接矩形
    roi = ref[y:y+h,x:x+w]
    roi = cv.resize(roi,(57,88))
    digits[i] = roi


image = cv.imread('project3_1.png')
image_gray= cv.cvtColor(image,cv.COLOR_BGR2GRAY)

#进行礼帽操作，把突出明亮区域
kernel = np.ones((9,9),np.uint8)
tophat = cv.morphologyEx(image_gray,cv.MORPH_TOPHAT,kernel)

#梯度运行(Sobel算子) dx=1说明只对x方向求导，ksize=-1说明使用Scharr算子
gradX = cv.Sobel(tophat,ddepth=cv.CV_32F,dx=1,dy=0,ksize=-1)
(minVal,maxVal) = (np.min(gradX),np.max(gradX))
gradX = np.absolute(gradX)
gradX = (255*((gradX-minVal)/(maxVal-minVal)))
gradX = gradX.astype('uint8') #转模式
#先膨胀再腐蚀
gradX = cv.morphologyEx(gradX,cv.MORPH_CLOSE,kernel)

#再进行阈值分割
thresh = cv.threshold(gradX,10,255,cv.THRESH_BINARY|cv.THRESH_OTSU)[1]

#再进行闭运算
gradX = cv.morphologyEx(thresh,cv.MORPH_CLOSE,kernel)

threshCnts,hierarchy = cv.findContours(gradX.copy(),cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

#遍历轮廓
locs = []
for (i,c) in enumerate(threshCnts):
    (x,y,w,h) = cv.boundingRect(c)
    ar = w/float(h)

    #选择合适的区域
    if ar>2.5 and ar<4.0:
        if (w>40 and w<60) :
            locs.append((x,y,w,h))

#将符合的轮廓从左到右进行排序
locs = sorted(locs,key = lambda x:x[0])

#遍历每一个轮廓中的数字
outPut = []
for (i,(gX,gY,gW,gH)) in enumerate(locs):
    #根据坐标提前每一个组
    group = image_gray[gY-5:gY+gH+5,gX-5:gX+gW+5]
    group = cv.threshold(group,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)[1]

    digitCnts,_= cv.findContours(group.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    digitCnts = sorted(digitCnts,key=lambda x:cv.boundingRect(x)[0])

    for c in digitCnts:
        (x,y,w,h) = cv.boundingRect(c)
        roi = group[y:y+h,x:x+w]
        roi = cv.resize(roi,(57,88))
        scores = []
        groupOutput = []
        #进行模版匹配
        for (digit,digitROI) in digits.items():
            '''进行模版匹配'''
            result = cv.matchTemplate(roi,digitROI,cv.TM_CCOEFF)
            (_,score,_,_) = cv.minMaxLoc(result)
            scores.append(score)
            groupOutput.append(digit)
        outPut.append(groupOutput[scores.index(max(scores))])

print(outPut)