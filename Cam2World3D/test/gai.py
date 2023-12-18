import numpy as np
import cv2

# 加载相机参数
def load_parameters():
    with np.load(r'full.npz') as data:
        mat_intri = data['mat_intri']
        coff_dis = data['coff_dis']
    return mat_intri, coff_dis

boxPoints = np.array(
    [
        [397, 356], [736, 362],  # 第一行
        [342, 680], [776, 694]  # 最后一行
    ],
    dtype=np.float32)

print("角点坐标:\n", boxPoints)

# 定义8个角点的世界坐标
worldBoxPoints = np.array(
    [
        [0, 0, 0], [0.43, 0, 0],  # 第一行
        [0, 0.5, 0], [0.43, 0.5, 0]  # 最后一行
    ],
    dtype=np.float32)

# 加载相机内参和畸变系数
cameraMatrix1, distCoeffs1 = load_parameters()
# print("相机内参:\n", cameraMatrix1)
print("畸变系数:\n", distCoeffs1)

ret, rvecs, tvecs, _ = cv2.solvePnPGeneric(worldBoxPoints, boxPoints, cameraMatrix1, distCoeffs1)
rvec = rvecs[0]
tvec = tvecs[0]

# 保存旋转向量和平移向量
np.savez('RT.npz', rvec=rvec, tvec=tvec)

# 使用projectPoints投影世界坐标到像素坐标
projected_points, _ = cv2.projectPoints(worldBoxPoints, rvec, tvec, cameraMatrix1, distCoeffs1)

for i, point in enumerate(projected_points):
    print(f"角点 {i + 1} 投影像素坐标:", point[0])

# 计算旋转矩阵
R, _ = cv2.Rodrigues(rvec)

# 已知的Z坐标
Zconst = 0

# 选择一个角点的索引（例如，第一个角点索引为0）
index = 0

# 获取2D像素坐标
uvPoint = np.array([boxPoints[index][0], boxPoints[index][1], 1])

# 计算s（按照原理形式）
leftMat = np.linalg.inv(R) @ np.linalg.inv(cameraMatrix1) @ uvPoint
rightMat = np.linalg.inv(R) @ tvec
s = (Zconst + rightMat[2]) / leftMat[2]

# 计算3D世界坐标（按照原理形式）
worldCoord = np.linalg.inv(R) @ (s * np.linalg.inv(cameraMatrix1) @ uvPoint - tvec)

# 将worldCoord转换为1x3向量
worldCoord = worldCoord.flatten()

# 打印输出
print(f"World Coordinates of corner at index {index}: x={worldCoord[0]}, y={worldCoord[1]}, z={worldCoord[2]}")
