import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('project2_1.png')
ratio = img.shape[0]/500
orig = img.copy()
img = cv.resize(img,(500,int(img.shape[0]/ratio)))

#转为灰度图像，并且进行高斯模糊
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray,(5,5),0)
#进行边缘检测
edges = cv.Canny(gray,75,220)
plt.axis('off')
plt.imshow(edges,cmap=plt.gray())
plt.show()

'''
找轮廓:
第二个参数:所有轮廓都要，但不构成层级关系，都是平级
第三个参数:不需要所有像素点，这个是压缩后的

'''
contours,_ = cv.findContours(edges,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
'''按照面积排序的，所以第一个是排序最大的，也就是外面的但是返回值是按照面积排序的，所以第一个是排序最大的，也就是外面的
'''
contours = sorted(contours,key=cv.contourArea,reverse=True)
biggset_cnt = contours[0]

#画出最大轮廓
cv.drawContours(img,[biggset_cnt],-1,(0,255,0),3)
plt.axis('off')
plt.imshow(img[:,:,::-1])
plt.show()

peri = cv.arcLength(biggset_cnt,True)
'''
它能把坑坑洼洼、由几百个点组成的曲线边缘，浓缩成几个关键的拐点。
为什么是周长的2%呢，根据工业的经验值算出来的
'''
approx = cv.approxPolyDP(biggset_cnt,0.02*peri,True)
if len(approx)==4:
    '''
    说明是个四边形
    那么给四个角排序(左上，右上,右下，左下)
    pts取出来的四个坐标不是按照左上，右上，右下，左下进行排列的，而是随机的这四个坐标
    sum(axis=1)就是对每一行进行求和，左上是最小的，右下是最大的，那么就可以取出来
    '''
    pts = approx.reshape(4,2)
    rect = np.zeros((4,2),dtype='float32')

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] #左上角
    rect[2] = pts[np.argmax(s)] #右上角

    '''
    diff = np.diff(pts, axis=1) 等价于 x - y，或者你可以直接写 pts[:, 0] - pts[:, 1]
    '''
    d = np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(d)] #右上角
    rect[3] = pts[np.argmax(d)] #左下角

    #进行透视变换
    (tl,tr,br,bl) = rect
    widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1]-bl[1])**2))
    widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1]-tl[1])**2))
    maxWidth = max(int(widthA),int(widthB))

    heightA = np.sqrt(((br[0] - bl[0])**2) + ((br[1]-bl[1])**2))
    heightB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1]-tl[1])**2))
    maxHeight = max(int(heightA),int(heightB))

    if maxWidth > maxHeight:
        maxWidth,maxHeight = maxHeight,maxWidth
    #定义变换后的四个目标点
    dst = np.array([
        [0,0],
        [maxWidth-1,0],
        [maxWidth-1,maxHeight-1],
        [0,maxHeight-1]],dtype='float32')

    #进行透视变换,先进行透视变换矩阵的计算
    M = cv.getPerspectiveTransform(rect,dst)
    warped = cv.warpPerspective(img,M,(maxWidth,maxHeight))
    warped = cv.rotate(warped,cv.ROTATE_90_COUNTERCLOCKWISE)
    plt.axis('off')
    plt.imshow(warped[:,:,::-1])
    plt.show()

    #转灰度
    warped_gray = cv.cvtColor(warped,cv.COLOR_BGR2GRAY)
    warped_blur = cv.GaussianBlur(warped_gray,(5,5),0)
    warped_thresh = cv.adaptiveThreshold(warped_blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,9,2)
    # ret,warped_thresh = cv.threshold(warped_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    plt.axis('off')
    plt.imshow(warped_thresh,cmap=plt.gray())
    plt.show()
else:
    pass