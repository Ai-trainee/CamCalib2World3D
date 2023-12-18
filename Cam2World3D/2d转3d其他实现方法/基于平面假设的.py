import cv2
import numpy as np

def load_parameters():
    with np.load('./25.npz') as data:
        mat_intri = data['mat_intri']
        coff_dis = data['coff_dis']
    print("加载的内参矩阵:\n", mat_intri)
    print("加载的畸变系数:\n", coff_dis)
    return mat_intri, coff_dis

def define_plane(worldBoxPoints):
    point1, point2, point3 = worldBoxPoints[:3, 0] # 注意这里的索引
    vector1 = point2 - point1
    vector2 = point3 - point1
    normal_vector = np.cross(vector1, vector2).flatten()
    D = -np.dot(normal_vector, point1.flatten())
    A, B, C = normal_vector
    return A, B, C, D



# def project_to_ground(boxPoints, cameraMatrix, distCoeffs, rvec, tvec, A, B, C, D):
#     ground_points = []
#     for point in boxPoints:
#         ray = cv2.undistortPoints(np.array([[[point[0], point[1]]]]), cameraMatrix, distCoeffs, P=cameraMatrix)
#         ray = cv2.transform(ray, np.linalg.inv(cv2.Rodrigues(rvec)[0]))
#         ray = ray[0, 0]
#         t = -(A * tvec[0] + B * tvec[1] + C * tvec[2] + D) / (A * ray[0] + B * ray[1] + C * ray[2])
#         X = ray * t + tvec
#         ground_points.append(X)
#     return np.array(ground_points)
# def project_to_ground(boxPoints, cameraMatrix, distCoeffs, rvec, tvec, A, B, C, D):
#     ground_points = []
#     for point in boxPoints:
#         ray = cv2.undistortPoints(np.array([[[point[0], point[1]]]]), cameraMatrix, distCoeffs, P=cameraMatrix)
#         ray = cv2.transform(ray, np.linalg.inv(cv2.Rodrigues(rvec)[0]))
#         ray = ray[0, 0]
#         t = -(A * tvec[0] + B * tvec[1] + C * tvec[2] + D) / (A * ray[0] + B * ray[1] + C * ray[2])
#         X = ray * t + tvec
#         ground_points.append(X)
#     ground_points = np.round(ground_points).astype(int) # 舍入并转换为整数
#     return ground_points
def project_to_ground(boxPoints, cameraMatrix, distCoeffs, rvec, tvec, A, B, C, D):
    ground_points = []
    for point in boxPoints:
        ray = cv2.undistortPoints(np.array([[[point[0], point[1]]]]), cameraMatrix, distCoeffs, P=cameraMatrix)
        ray = cv2.transform(ray, np.linalg.inv(cv2.Rodrigues(rvec)[0]))
        ray = ray[0, 0]
        # print("Ray:", ray) # 打印 ray
        t = -(A * tvec[0] + B * tvec[1] + C * tvec[2] + D) / (A * ray[0] + B * ray[1] + C * ray[2])
        X = (ray * t + tvec.T).flatten() # 注意这里使用了 tvec.T
        # print("X:", X) # 打印 X
        ground_points.append(X)
    ground_points = np.array(ground_points)
    return ground_points



def main():
    sourceImage = cv2.imread(r"D:\Desktop\Calibration-ZhangZhengyou-Method-master\picss\IMG_20230815_160013.jpg")
    gray = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)
    pattern_size = (8, 6)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        print("无法找到棋盘格角点，请检查图像和棋盘格大小。")
        return



    cameraMatrix, distCoeffs = load_parameters()
    # 按顺时针方向排列四个角点
    boxPoints = np.array([corners[0][0], corners[pattern_size[0] - 1][0], corners[-1][0], corners[-pattern_size[0]][0]],
                         dtype=np.float32)
    print("使用的四个角点像素坐标:\n", boxPoints)

    worldBoxPoints = np.array([[[0, 0, 0]], [[20, 0, 0]], [[20, 14, 0]], [[0, 14, 0]]], dtype=np.float32)  # 世界坐标
    _, rvec, tvec, inliers = cv2.solvePnPRansac(worldBoxPoints, boxPoints, cameraMatrix, distCoeffs)

    print("使用的世界坐标点:\n", worldBoxPoints)
    # print("旋转向量:\n", rvec)
    # print("平移向量:\n", tvec)
    A, B, C, D = define_plane(worldBoxPoints)
    ground_points = project_to_ground(boxPoints, cameraMatrix, distCoeffs, rvec, tvec, A, B, C, D)
    ground_points_int = ground_points.astype(int)  # 转换为整数类型
    print("投影到地面的点（整数坐标）:\n", ground_points_int)


if __name__ == "__main__":
    main()
