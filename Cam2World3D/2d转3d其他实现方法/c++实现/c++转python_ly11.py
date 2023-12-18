import cv2
import numpy as np
import os

# 读取文件夹中的图片
image_folder = r"D:\Desktop\Calibration-ZhangZhengyou-Method-master\pic\RGB_camera_calib_img" # 请替换为您的文件夹路径
images = [cv2.imread(os.path.join(image_folder, img_name)) for img_name in os.listdir(image_folder) if img_name.endswith(('.png', '.jpg', '.jpeg'))]

# 获取图像大小
image_size = images[0].shape[:2]
board_size =(11,8)
corners_pixel = []

# 寻找并提取棋盘角点
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_points = cv2.findChessboardCorners(gray, board_size)
    if ret:
        cv2.cornerSubPix(gray, img_points, (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        corners_pixel.append(img_points)

        # 在图像上绘制棋盘角点
        cv2.drawChessboardCorners(img, board_size, img_points, ret)

        # 显示图像
        cv2.imshow('Chessboard Corners', img)


# 设置棋盘格子大小，并计算真实的点坐标
square_size = (2, 2)
corners_space = [[(k * square_size[0], j * square_size[1], 0) for j in range(board_size[1]) for k in range(board_size[0])] for _ in images]

# 相机标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(corners_space, corners_pixel, image_size[::-1], None, None)

# 使用RANSAC算法求解旋转和平移矩阵
RRvecs = [cv2.Rodrigues(rvec_ransac)[0] for _, rvec_ransac, _ in [cv2.solvePnPRansac(corners_space[i], corners_pixel[i], camera_matrix, dist_coeffs) for i in range(len(images))]]

# 计算投影矩阵
outcan = np.hstack((RRvecs[0], tvecs[0]))
Fin = camera_matrix.dot(outcan)
Inv = np.linalg.inv(Fin)

# 计算相机矩阵
zero = np.zeros((3, 1))
cam = np.hstack((camera_matrix, zero))
outcan2 = np.vstack((outcan, [0, 0, 0, 1]))
zeri = np.array([[0, 0, 0, 1]]).T
sa = cam.dot(outcan2).dot(zeri)
s = sa[2, 0]
a = sa / s

# 输入像素坐标
pixel_x = int(input("请输入像素坐标x："))
pixel_y = int(input("请输入像素坐标y："))
pixel_xx = 320 - pixel_x + a[0, 0]
pixel_yy = 240 - pixel_y + a[1, 0]
pro_pixel = np.array([[pixel_xx, pixel_yy, 1]]).T

# 计算真实坐标
real_coordinates = 163.84 * s * Inv.dot(pro_pixel)

# 输出真实坐标
print(f"真实坐标：X = {real_coordinates[0, 0]}, Y = {real_coordinates[1, 0]}")
