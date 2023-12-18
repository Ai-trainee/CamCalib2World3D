import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_parameters():
    with np.load(r'D:\Desktop\Calibration-ZhangZhengyou-Method-master\图像坐标转3d\now\4k.npz') as data:
        mat_intri = data['mat_intri']
        coff_dis = data['coff_dis']
    return mat_intri, coff_dis

# 读取图像
sourceImage = cv2.imread(r"ar3.png")
gray = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)
if gray is None:
    print("无法读取图像，请检查文件路径和完整性。")
    exit()

import cv2 as cv
import numpy as np

dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
parameters = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(dictionary, parameters)

frame = cv.imread(r"ar3.png")  # 请确保这个路径是正确的
markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

# print("检测到的标记角点：")
# print(markerCorners)
# print("检测到的标记ID：")
# print(markerIds)

cv.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

# 创建一个包含（角点，ID）对的列表
markers_with_ids = list(zip(markerCorners, markerIds))

# 按照ID排序
sorted_markers_with_ids = sorted(markers_with_ids, key=lambda x: x[1][0])
all_top_left_corners = []
# 输出排序后的每个标记的左上角像素角点
print("按ID排序后的左上角像素角点：")
for corners, id in sorted_markers_with_ids:
    top_left_corner = corners[0][0].astype(int)  # 左上角是第一个角点
    print(f"ID {id[0]} 的左上角像素角点：{tuple(top_left_corner)}")
    all_top_left_corners.append(top_left_corner)
# 将all_top_left_corners转换为NumPy数组，并赋值给boxPoints
boxPoints = np.array(all_top_left_corners, dtype=np.float32)
print("boxPoints:\n", boxPoints)

# 取出排序后的第一个标记的角点和ID
first_marker_corners, first_marker_id = sorted_markers_with_ids[0]

print(f"第一个被检测到的标记的ID是：{first_marker_id[0]}")

for i, corner in enumerate(first_marker_corners[0]):
    cv.circle(frame, tuple(corner.astype(int)), 5, (0, 255, 0), 2)
    cv.putText(frame, str(i + 1), tuple(corner.astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 打印第一个角点像素坐标
    if i == 0:
        print("第一个角点像素坐标：")
        print(tuple(corner.astype(int)))

    # 打印所有角点像素坐标
    print(f"第{i + 1}个角点像素坐标：")
    print(tuple(corner.astype(int)))

# 标记像素坐标xy朝向
cv.arrowedLine(frame, (50, 50), (80, 50), (0, 0, 255), 2, tipLength=0.3)  # x轴
cv.arrowedLine(frame, (50, 50), (50, 80), (0, 0, 255), 2, tipLength=0.3)  # y轴
cv.putText(frame, "x", (90, 55), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv.putText(frame, "y", (45, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# 缩放图像
scale_percent = 120  # 缩放到原来的120%
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
resized_frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)

cv.imshow("frame", resized_frame)
cv.waitKey(0)
cv.destroyAllWindows()

# boxPoints = np.array([[1111., 3.],
#                       [745., 24.],
#                       [11., 552.],
#                       [730., 567.]])
# print("boxPoints2:\n", boxPoints)













# 以下是你原有的代码，我没有进行任何修改
worldBoxPoints = np.array([[0, 0, 0], [0.14, 0, 0], [0, 0.10, 0], [0.14, 0.10, 0]], dtype=np.float32)

cameraMatrix1, distCoeffs1 = load_parameters()
ret, rvec, tvec = cv2.solvePnP(worldBoxPoints, boxPoints, cameraMatrix1, distCoeffs1)
R, _ = cv2.Rodrigues(rvec)
projected_points, _ = cv2.projectPoints(worldBoxPoints, rvec, tvec, cameraMatrix1, distCoeffs1)

for i, point in enumerate(projected_points):
    print(f"角点 {i + 1} 投影像素坐标:", point[0])

# 使用第一个方法的数学公式
for index in range(4):  # 循环处理四个角点
    pixel_point = boxPoints[index]
    homogeneous_pixel = np.array([pixel_point[0], pixel_point[1], 1])

    ray_dir = np.linalg.inv(cameraMatrix1).dot(homogeneous_pixel)
    ray_dir = R.T.dot(ray_dir)
    t = -tvec[2] / (R.T[2].dot(ray_dir))
    world_point = -t * ray_dir - R.T.dot(tvec).flatten()

    X, Y, Z = world_point
    print(f"角点 {index + 1} 的射线与地面交点的世界坐标: X={X}, Y={Y}, Z={Z}")

    # 在原图像上标注射线与地面交点
    cv2.circle(sourceImage, (int(pixel_point[0]), int(pixel_point[1])), 5, (255, 0, 0), -1)
    cv2.putText(sourceImage, f"X={X:.2f}, Y={Y:.2f}", (int(pixel_point[0]), int(pixel_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# 缩放图像
scale_percent = 60  # 缩放到原来的60%
width = int(sourceImage.shape[1] * scale_percent / 100)
height = int(sourceImage.shape[0] * scale_percent / 100)
dim = (width, height)
resized_frame = cv2.resize(sourceImage, dim, interpolation=cv2.INTER_AREA)

cv2.imshow('Image with Points', resized_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
