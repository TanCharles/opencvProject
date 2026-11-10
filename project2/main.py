import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像并转灰度
img = cv2.imread('project1_4.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 强力去噪（必须！否则Canny全是碎线）
# 这里可以用高斯模糊，甚至双边滤波（BilateralFilter）保留边缘的同时去噪
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. Canny 边缘检测
# 阈值设低一点，确保暗处的螺丝边缘也能被提取出来
edges = cv2.Canny(blur, 30, 100)
plt.axis('off')
plt.imshow(edges,cmap=plt.gray())
plt.show()

# 4. 🌟 核心绝招：暴力闭运算（用大核把边缘连起来，并填实内部）
# 因为线条是散的，用 15x15 甚至 21x21 的大核，强行让线条膨胀并粘连在一起
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
plt.axis('off')
plt.imshow(closed_edges,cmap=plt.gray())
plt.show()
# 5. 找轮廓 (只找最外层)
contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 6. 画图过滤
img_result = img.copy()
valid_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)

    # 过滤掉面积太小的噪点
    if area > 200:
        valid_count += 1
        # 计算最小外接矩形
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_result, [box], 0, (0, 255, 0), 2)

print(f"✅ Canny方案检测完成，找到 {valid_count} 个目标")
plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()