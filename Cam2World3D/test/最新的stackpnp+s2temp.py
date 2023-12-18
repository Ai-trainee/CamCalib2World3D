import cv2
import numpy as np

def load_parameters():
    with np.load(r'F:\920yijia3072/920yijia3072.npz') as data:
        mat_intri = data['mat_intri']
        coff_dis = data['coff_dis']
    return mat_intri, coff_dis

# 读取图像
sourceImage = cv2.imread(r"F:\920yijia3072\2.jpg")
sourceImage = cv2.imread(r"F:\920yijia3072\gaokong\5.jpg")
gray = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)

# 检测棋盘格角点
pattern_size = (8, 6)
found, corners = cv2.findChessboardCorners(gray, pattern_size)
if not found:
    print("无法找到棋盘格角点，请检查图像和棋盘格大小。")
else:
    # 按第一行两个和第二行两个点的顺序排列四个角点
    boxPoints = np.array(
        [corners[0][0], corners[pattern_size[0] - 1][0], corners[-pattern_size[0]][0], corners[-1][0]],
        dtype=np.float32)
    # 在图像上标识这四个角点
    for i, corner in enumerate(boxPoints):
        x, y = tuple(corner)
        cv2.circle(sourceImage, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(sourceImage, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    print("角点坐标:\n", boxPoints)
    # 世界坐标
    worldBoxPoints = np.array([[0, 0, 0], [0.14, 0, 0], [0, 0.10, 0], [0.14, 0.10, 0]], dtype=np.float32)  # 世界坐标



    # 加载相机内参和畸变系数
    cameraMatrix1, distCoeffs1 = load_parameters()
    print("相机内参:\n", cameraMatrix1)
    print("畸变系数:\n", distCoeffs1)

    # 使用solvePnP求取外参
    ret, rvec, tvec = cv2.solvePnP(worldBoxPoints, boxPoints, cameraMatrix1, distCoeffs1)
    print("旋转向量:\n", rvec)
    print("平移向量:\n", tvec)
    #保存旋转向量和平移向量
    np.savez('../图像坐标转3d/now/RT.npz', rvec=rvec, tvec=tvec)

    # 使用projectPoints投影世界坐标到像素坐标
    projected_points, _ = cv2.projectPoints(worldBoxPoints, rvec, tvec, cameraMatrix1, distCoeffs1)

    for i, point in enumerate(projected_points):
        print(f"角点 {i + 1} 投影像素坐标:", point[0])

    # 计算旋转矩阵
    R, _ = cv2.Rodrigues(rvec)

    # 已知的Z坐标
    Zconst = 0

    # 选择一个角点的索引（例如，第一个角点索引为0）
    index = 3

    # 获取2D像素坐标
    uvPoint = np.array([boxPoints[index][0], boxPoints[index][1], 1])

    # 计算s（按照原理形式）
    tempMat = np.linalg.inv(np.dot(cameraMatrix1, R) - uvPoint.reshape(-1, 1) @ tvec.reshape(1, -1))
    s = Zconst + np.dot(tempMat[-1, :], tvec)
    s /= np.dot(tempMat[-1, :], uvPoint)

    # 计算3D世界坐标（按照原理形式）
    worldCoord = np.dot(R.T, (np.linalg.inv(cameraMatrix1) @ (s * uvPoint) - tvec))

    # 将worldCoord转换为1x3向量
    worldCoord = worldCoord.flatten()
    print("世界坐标:\n", worldBoxPoints)
    # 打印输出
    print(f"World Coordinates of corner at index {index}: x={worldCoord[0]}, y={worldCoord[1]}, z={worldCoord[2]}")



