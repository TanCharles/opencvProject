import os
import cv2 as cv
import numpy as np



#获取所有图像内容
def get_images_by_dir(dirname):
    img_names = os.listdir(dirname)
    img_paths = [os.path.join(dirname,img_name) for img_name in img_names]
    imgs = [ cv.imread(img_path) for img_path in img_paths]
    return imgs

#获取棋盘格点的现实坐标与像素点下的坐标
def calibrate(images,grid=(9,6)):
    object_points = [] #真实世界的值
    img_points = []  #像素点坐标系里面的值

    for img in images:
        '''
        np.mgrid其实是生成现实坐标系中的值，为什么不用尺量出来呢
        因为尺量出来，每个点也是(0,0,0) (0,25,0) (0,50,0) 
        和生成出来的值(0,0,0) (0,1,0) (0,2,0) 计算出来的相机内参是不会变的
        '''
        object_point = np.zeros((grid[0]*grid[1],3),np.float32)
        object_point[:,:2] = np.mgrid[0:grid[0],0:grid[1]].T.reshape(-1,2)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret,corners = cv.findChessboardCorners(gray,grid,None)
        if ret:
            object_points.append(object_point)
            img_points.append(corners)

    return object_points,img_points


#把输入图像去除相机畸变，生成恢复后的图像
def cal_undistort(img,objpoints,imgpoints):
    '''
    ret:衡量判定好坏的分数
    mtx:内参
    dist:畸变参数
    rvecs,tvecs:外参 (旋转和平移)
    '''
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    dst = cv.undistort(img, mtx, dist, None, mtx)
    return dst


#进行透视变换，得到变换矩阵和逆变换矩阵(还需要把鸟瞰图加到原图里面去)
def get_M_Minv():
    src = np.float32([[(203,720),(585,460),(695,460),(1127,720)]])
    dst = np.float32([[(320,720),(320,0),(960,0),(960,720)]])
    M = cv.getPerspectiveTransform(src, dst)
    Minv = cv.getPerspectiveTransform(dst,src)
    return M,Minv


#获取图像x方向或者y方向的边缘检测(在这个阈值范围内的，就设置为1，否则设置为0,这个只会检测水平方向的)
def abs_sobel_thresh(img,orient='x',thresh_min=0,thresh_max=255):
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv.Sobel(gray,cv.CV_64F,1,0))
    if orient == 'y':
        abs_sobel = np.absolute(cv.Sobel(gray,cv.CV_64F,0,1))


    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[((scaled_sobel>=thresh_min) & (scaled_sobel<=thresh_max))] =1
    return binary_output


#这个不管什么方向，都会进行检测，如果在弯道方向，这个算法比上面那个算法好
def mag_thresh(img,sobel_kernel=3,mag_thresh=(0,255)):
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    sobelx = cv.Sobel(gray,cv.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv.Sobel(gray,cv.CV_64F,0,1,ksize=sobel_kernel)

    gradmag = np.sqrt(sobelx**2+sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[((gradmag>=mag_thresh[0]) & (gradmag <= mag_thresh[1]))] = 1
    return binary_output


#这个是选择相应角度的线条，斜的但是偏水平的，就是远处的车道线
def dir_threshold(img,sobel_kernel=3,dir_thresh=(0,np.pi/2)):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)

    absgraddir = np.arctan2(np.absolute(sobely),np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[((absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1]))] = 1
    return binary_output


#黄色的车道线，S饱和度比较高，白色的车道线L亮度很高，就可以区分
def hls_select(img,channel='s',thresh=(0,255)):
    hls = cv.cvtColor(img,cv.COLOR_RGB2HLS)
    if channel=='h':
        channel = hls[:,:,0]
    elif channel=='l':
        channel = hls[:,:,1]
    else:
        channel = hls[:,:,2]

    binary_output = np.zeros_like(channel)
    binary_output[(channel>thresh[0]) & (channel<=thresh[1])] = 1
    return binary_output

# L 亮度:0黑~100白 a: a负=绿，a正=红 b:b负=蓝 b正=黄
def lab_select(img,thresh=(0,255)):
    lab = cv.cvtColor(img,cv.COLOR_RGB2LAB)
    b_channel = lab[:,:,2]
    binary_output = np.zeros_like(b_channel)
    binary_output[(b_channel>thresh[0]) & (b_channel<=thresh[1])] = 1
    return binary_output

def luv_thresh(img,thresh=(0,255)):
    luv = cv.cvtColor(img,cv.COLOR_RGB2LUV)
    l_channel = luv[:,:,0]
    binary_out_put = np.zeros_like(l_channel)
    binary_out_put[(l_channel>thresh[0]) & (l_channel<=thresh[1])] = 1
    return binary_out_put


def find_line(binary_warped):
    #对图像下半部分绘制直方图
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:],axis=0)

    midpoint = np.int32(histogram.shape[0]/2)
    #返回的最大值的索引
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 9
    window_height = np.int32(binary_warped.shape[0]/nwindows)
    #取出所有不为0的像素点的坐标，返回是一个元组 nonzero[0] --> y(行下标)  nonzero[1] --->x(列下标)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    #框的宽度的一半
    margin = 100
    #最小的白色像素点
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1) * window_height
        win_y_high = binary_warped.shape[0] - window*window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        #筛选在这个窗口内的所有白色点的在nonzero 中的下标
        good_left_inds = ((nonzeroy>=win_y_low) & (nonzeroy<win_y_high) &
                          (nonzerox>=win_xleft_low) & (nonzerox<win_xleft_high)).nonzero()[0]

        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if (len(good_left_inds)) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))

        if (len(good_right_inds)) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    #把符号条件的nonzeros 下标的数组 拼接在一起，成为一个一维数组
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    #拟合成曲线: x=a*y^2+b*y+c 返回的是[a,b,c]
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)

    return left_fit,right_fit,left_lane_inds,right_lane_inds


def find_line_previous(binary_warped,left_line,right_line):
    nonzero = binary_warped.nonzero()

    nonzerox = np.array(nonzero[0])
    nonzeroy = np.array(nonzero[1])

    margin=100
    left_fit = left_line.current_fit
    right_fit = right_line.current_fit
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) +left_fit[1]*nonzeroy + left_fit[2] -margin)) &
                      (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy+left_fit[2]+margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
                      (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # ========== 新增判断 ==========
    min_points = 10  # 最少需要多少个点才允许拟合，可自己调
    left_valid = len(leftx) > min_points
    right_valid = len(rightx) > min_points

    # 这里你要从 left_line / right_line 拿到上一帧的 fit
    left_fit_prev = left_line.best_fit
    right_fit_prev = right_line.best_fit

    if left_valid:
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        left_fit = left_fit_prev  # 点不够，沿用之前的拟合
    if right_valid:
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        right_fit = right_fit_prev  # 点不够，沿用之前的拟合

    return left_fit,right_fit,left_lane_inds,right_lane_inds

step=0
def draw_area(undist,binary_warped,Minv,left_fit,right_fit):
    #**在 start ~ stop 之间，均匀生成 num 个数字**，返回一个一维数组。
    '''
    hstack-->水平拼接(左右拼接)
    vstack-->上下拼接
    dstack-->深度拼接
    '''
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # 注意：这里不要外面那一层 [] ！得到 (N,2)，不是 (1,N,2)
    pts_left = np.transpose(np.vstack([left_fitx, ploty]))
    pts_right = np.transpose(np.vstack([right_fitx, ploty]))

    # 现在pts_right是(N,2)，flipud可以正常翻转，点从下到上
    pts = np.vstack((pts_left, np.flipud(pts_right)))

    # fillPoly输入需要 [pts]，变成 (1,N,2)
    cv.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
    global step
    if step == 63:
        cv.imshow('r', binary_warped * 255)
        cv.waitKey(0)
        cv.imshow('w',color_warp)
        cv.waitKey(0)
    step += 1
    newwarp = cv.warpPerspective(color_warp,Minv,(undist.shape[1],undist.shape[0]))
    #undist 原图 alpha:1 原图权重  newwarp:第二张图, beta:0.3(第二张图的权重)
    result = cv.addWeighted(undist,1,newwarp,0.3,0)
    return result

