import cv2 as cv

dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
parameters = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(dictionary, parameters)

frame = cv.imread(r"2023-09-19/192.168.1.40_01_20230919101813295.jpg")  # 请确保这个路径是正确的
markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

print("检测到的标记角点：")
print(markerCorners)
print("检测到的标记ID：")
print(markerIds)

cv.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

# 找到ID为0的标记的索引
index_of_id_0 = None
for i, id in enumerate(markerIds):
    if id[0] == 0:
        index_of_id_0 = i
        break

# 如果找到了ID为0的标记，标记其角点
if len(markerCorners) > 0 and len(markerIds) > 0:
    first_detected_corners = markerCorners[0][0]
    first_detected_id = markerIds[0][0]

    print(f"第一个被检测到的标记的ID是：{first_detected_id}")

    for i, corner in enumerate(first_detected_corners):
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
scale_percent = 120  # 缩放到原来的200%
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
resized_frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)

cv.imshow("frame", resized_frame)
cv.waitKey(0)
cv.destroyAllWindows()
