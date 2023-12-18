import cv2
import numpy as np

def load_parameters():
    with np.load(r'D:\Desktop\Calibration-ZhangZhengyou-Method-master\Fhaikang2.npz') as data:
        mat_intri = data['mat_intri']
        coff_dis = data['coff_dis']
    return mat_intri, coff_dis

# 读取图像
sourceImage = cv2.imread(r"F:\haikang1\7.png")
gray = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)

# 检测棋盘格角点
pattern_size = (8, 6)
found, corners = cv2.findChessboardCorners(gray, pattern_size)
if not found:
    print("无法找到棋盘格角点，请检查图像和棋盘格大小。")
else:
    #角点检测顺序为每一行从左到右
    boxPoints = np.array(
        [corners[0][0], corners[pattern_size[0] - 1][0], corners[-pattern_size[0]][0], corners[-1][0]],
        dtype=np.float32)

    for i, corner in enumerate(boxPoints):
        x, y = tuple(corner)
        cv2.circle(sourceImage, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(sourceImage, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #世界坐标定义和像素坐标一样，顺序为每一行从左到右，单位0.02m
    worldBoxPoints = np.array([[0, 0, 0], [0.14, 0, 0], [0, 0.10, 0], [0.14, 0.10, 0]], dtype=np.float32)
    print("世界坐标:\n", worldBoxPoints)

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


