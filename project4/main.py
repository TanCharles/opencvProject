import os
import line
import utils
import cv2 as cv
import numpy as np
from moviepy.editor import VideoFileClip
'''
1. x_thresh（abs_sobel_thresh x 方向）
    - 作用：提取**横向亮度变化大的边缘**，车道线是纵向长条，左右两边有明显明暗跳变，Sobel X 对车道竖边缘很敏感
    - 参数：`thresh_min=10, thresh_max=230`：梯度不能太弱（噪声）、也不能爆炸（强光白斑）
    - 缺点：
        路面裂纹、污渍、接缝、路边杂草、阴影边缘，同样会产生 x 方向梯度，**单独用噪点巨多**

2. mag_thresh 梯度幅值（sobel_kernel=3，(30,150)）
    - 作用：约束**边缘强度**，只有梯度大小在区间内才算有效边缘
    - 为什么和 x_thresh 做`&`（同时满足）：
    只靠 sobel x：有些微弱的纹理（细小石子、细纹）也会有一点点 x 梯度，但整体幅值很小，直接过滤掉；
    逻辑含义：**有横向边缘，并且这个边缘的强度足够，不是微弱噪声**。
    ✅ 分支 A(1与上2) 适合：**明暗对比很强、清晰的白色车道线边缘**，靠梯度边缘抓线。
    ❌ 不擅长：发黄褪色车道线、强光下饱和度被冲没、阴影里梯度被压低的线。
    
3. dir_thresh 梯度方向阈值 (0.7 ~1.3 弧度)
    换算角度：0.7rad≈40°，1.3rad≈74°
    这个角度算出来的梯度方向的，梯度方向与边缘是垂直的，这个角度是梯度方向与x轴的夹角。
    车道线在图像里一般是**接近竖直、带一点倾斜**的长条，它的边缘梯度方向落在这个角度区间；
    - 作用：过滤水平方向的边缘！比如路面横向接缝、横向阴影、斑马线横条（它们梯度方向接近 0/90 以外）
    - 缺点：很多高饱和杂物（路边黄色警示牌、黄色污渍）饱和度很高，方向也可能刚好命中，单独用容易误检
    

4. hls_thresh：HLS 空间 S 通道（饱和度，180~255）
    H：色相，L：亮度，S：饱和度。
    - S 通道代表颜色纯不纯：黄色车道线饱和度很高；发白、灰色路面饱和度低
    - 作用：**抓高饱和度的黄色物体（黄车道线）**
    - 缺陷：强光暴晒下，黄色饱和度会被冲淡；白色车道饱和度很低，这个通道对白线无效

    分支B(3和4相与)：>> 像素是【高饱和度】+【边缘倾斜角度符合车道线】，才判定为车道。
    ✅ 分支 B 适合：**环境正常、饱和度保留较好的黄色车道线**，靠颜色 + 边缘角度双重校验。
    ❌ 不擅长：白线、褪色黄线、强光下饱和度丢失、阴影区。


5 lab_thresh：LAB 色彩空间 B 通道（155~200）
    LAB：
        - L：亮度
        - A：绿↔红
        - B：蓝↔黄
        **B 通道对黄色极其敏感，对白线不敏感**
    优势：不受饱和度爆炸的强光影响，比 HLS 的 S 通道稳定很多，逆光、树荫下抓黄线效果比 HLS 更好。
    ✅ 专门用来捡：阴影里、光照不均的**黄色车道线**

6 v_thresh：LUV 色彩空间 L 通道（225~255）
    LUV 的 L 通道是亮度通道，对**高亮度白色区域**非常友好
    专门抓**白色车道线**；白色在饱和度通道 (S) 几乎是 0，HLS 抓不到白线，靠 LUV-L。
    > > 为什么不用 RGB 灰度？RGB 灰度容易被路面亮斑、反光干扰；LUV 的 L 经过色度去耦合，对白线鲁棒性更强。
    这里是 `|` 或：**只要是 LAB 抓到的黄线 或者 LUV 抓到的白线，直接放行，不需要梯度条件！**
    这一点非常关键：
        很多时候车道线**边缘很模糊、磨损严重，梯度几乎看不见（A、B 分支捞不出来），但是颜色 / 亮度特征还在**，就靠 C 分支兜底。
    ✅ 分支 C 适合：磨损模糊车道线、梯度弱、边缘不清晰的黄线 / 白线，纯靠颜色 / 亮度特征。
    ❌ 风险：路面白斑、黄色污渍、路标色块，会被误触发，所以 C 分支是独立的一路，和梯度分支互补。
    

7: 边缘检测出来之后，采用透视变换，获得车道线的鸟瞰图
8: 采用二次项拟合的方法，去逼近那一条车道线边缘的曲线
    1)首先截图图像的下部分的图像，然后计算每一列的像素值的和
    2)然后分成左右两部分；分别选择在各自的区域内像素值的和最高的那一列作为横坐标
    3)把整个图像的高分为9个格子的高度，根据第三步选取出来的x，和 格子的高度 y,可以确定一个坐标（x,y）
    4)在这个坐标附近，进行框一个50*50大小的区域，然后统计里面不为0的像素点有多少个
    5)如果大于设置的个数，就把这个框内的像素的坐标加入到队列当中去，然后又重新算出平均的x坐标
    6)从下往上依次进行4-6的步骤
    7)根据队列里面的像素点的坐标，去拟合一条曲线,最后拿到相应的参数a,b,c
9：根据拿到的曲线的参数，去画出相应的两条曲线，然后中间进行填充即可。
    
'''


def thresholding(img):
    '''
    1. **HSV**：圆柱模型，H 色相，S 饱和度，V=RGB 最大值。适合简单调色、可控光照下颜色筛选；强光容易饱和糊掉。
    2. **HLS（HSL）**：双圆锥，H 色相，L 是平均亮度，S 饱和度。**车道线首选**，白线在 L 通道很好提取。
    3. **LAB（CIELAB）**：感知均匀，L 亮度，a 绿红，b 蓝黄；面向**反射表面**，光照鲁棒，工业色差、物体颜色分割。
    '''
    x_thresh = utils.abs_sobel_thresh(img,orient='x',thresh_min=10,thresh_max=230)
    mag_thresh = utils.mag_thresh(img,sobel_kernel=3,mag_thresh=(30,150))
    dir_thresh = utils.dir_threshold(img,sobel_kernel=3,dir_thresh=(0.7,1.3))
    hls_thresh = utils.hls_select(img,thresh=(180,255))
    lab_thresh = utils.lab_select(img,thresh=(155,200))
    luv_thresh = utils.luv_thresh(img,thresh=(225,255))
    thresholded = np.zeros_like(x_thresh)
    thresholded[((x_thresh == 1) &  (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | ((lab_thresh == 1) | luv_thresh==1)] = 1
    return thresholded

def processing(img,object_points,img_points,M,Minv,left_line,right_line):
    undist = utils.cal_undistort(img,object_points,img_points)
    thresholded = thresholding(undist)

    #是一个透视变换，采用的是双线插值法
    thresholded_wraped = cv.warpPerspective(thresholded,M,img.shape[1::-1],flags=cv.INTER_LINEAR)

    #得到一个x = a*y^2 + b*y + c 一个二次项拟合的公式，这条曲线就是车道线在透视图上的轨迹
    if left_line.detected and right_line.detected and False:
        left_fit, right_fit, left_lane_inds, right_lane_inds = utils.find_line_previous(thresholded_wraped,left_line, right_line)
    else:
        left_fit,right_fit,left_lane_inds,right_lane_inds = utils.find_line(thresholded_wraped)
    left_line.update(left_fit)
    right_line.update(right_fit)

    area_img = utils.draw_area(undist,thresholded_wraped,Minv,left_fit,right_fit)
    return area_img

if __name__ == '__main__':

    left_line = line.Line()
    right_line = line.Line()

    #获取相机标定的目录和像素点
    cal_imgs = utils.get_images_by_dir('cameraCal')
    object_points,img_points = utils.calibrate(cal_imgs,grid=(9,6))
    M,Minv = utils.get_M_Minv()


    project_outpath = 'project_video_out.mp4'
    # 生成一个视频剪辑对象
    project_video_clip = VideoFileClip('project_video.mp4')
    project_video_out_clip = project_video_clip.fl_image(
        lambda frame: processing(frame, object_points, img_points, M, Minv, left_line, right_line))
    project_video_out_clip.write_videofile(project_outpath, audio=False)