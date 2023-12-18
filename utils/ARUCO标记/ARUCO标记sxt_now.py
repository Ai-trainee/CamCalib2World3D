import cv2 as cv
import numpy as np

dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
parameters = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(dictionary, parameters)

frame = cv.imread(r"2023-09-19/192.168.1.40_01_20230919105234233.jpg")  # 请确保这个路径是正确的
markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

print("检测到的标记角点：")
print(markerCorners)
print("检测到的标记ID：")
print(markerIds)

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
scale_percent = 100  # 缩放到原来的120%
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
resized_frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)

cv.imshow("frame", resized_frame)
cv.waitKey(0)
cv.destroyAllWindows()
