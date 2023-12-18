import cv2
import numpy as np
def load_parameters():
    with np.load(r'C:\Users\25451\AppData\Roaming\JetBrains\PyCharm2023.1\scratches\full\1580\full1580.npz') as data:
        mat_intri = data['mat_intri']
        coff_dis = data['coff_dis']
    return mat_intri, coff_dis

def main():
    # 读取图像
    sourceImage = cv2.imread(
        r"C:\Users\25451\AppData\Roaming\JetBrains\PyCharm2023.1\scratches\full\1580\IMG20230830194613.jpg")
    gray = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)

    # 检测棋盘格角点
    pattern_size = (8, 6) # 根据您的棋盘格调整大小
    # pattern_size = (11,8)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        print("无法找到棋盘格角点，请检查图像和棋盘格大小。")
        return

    # 按顺时针方向排列四个角点
    boxPoints = np.array([corners[0][0], corners[pattern_size[0]-1][0], corners[-1][0], corners[-pattern_size[0]][0]], dtype=np.float32)
    print("角点坐标:\n",boxPoints)
    #
    worldBoxPoints = np.array([[0, 0, 0], [20, 0, 0], [20, 14, 0], [0, 14, 0]], dtype=np.float32)  # 世界坐标
    # 定义世界坐标，只包括XY平面坐标
    # worldBoxPoints = np.array([[0, 0], [20, 0], [20, 14], [0, 14]], dtype=np.float32)
    #打印世界坐标
    print("世界坐标:\n",worldBoxPoints)


    #相机内参矩阵 与 畸变系数
    cameraMatrix1, distCoeffs1 = load_parameters()
    # 使用cv2.solvePnPRansac得到初始解
    _, rvec1, tvec1, inliers = cv2.solvePnPRansac(worldBoxPoints, boxPoints, cameraMatrix1, distCoeffs1)

    # 使用cv2.solvePnPRefineVVS来精炼解
    rvec_refined, tvec_refined = cv2.solvePnPRefineVVS(worldBoxPoints, boxPoints, cameraMatrix1, distCoeffs1, rvec1,
                                                       tvec1)

    # 使用rvec_refined和tvec_refined替换原来的rvec1和tvec1
    rvec1 = rvec_refined
    tvec1 = tvec_refined

    # 以下代码保持不变
    rvecM1, _ = cv2.Rodrigues(rvec1)

    # 计算旋转角度
    thetaZ = np.arctan2(rvecM1[1, 0], rvecM1[0, 0]) / np.pi * 180
    thetaY = np.arctan2(-1 * rvecM1[2, 0], np.sqrt(rvecM1[2, 1]**2 + rvecM1[2, 2]**2)) / np.pi * 180
    thetaX = np.arctan2(rvecM1[2, 1], rvecM1[2, 2]) / np.pi * 180
    # print("theta x:", thetaX, "theta Y:", thetaY, "theta Z:", thetaZ)



    # 计算参数s (深度)
    index = 0  # 这里的序号是0，因为您正在使用boxPoints的第一个元素
    imagePoint = np.array([*boxPoints[index], 1], dtype=np.float64)  # 使用第一个boxPoints作为图像点
    print(f"测试角点序号 {index}:", imagePoint)
    zConst = 0  # 实际坐标系的距离
    tempMat = np.linalg.inv(rvecM1).dot(np.linalg.inv(cameraMatrix1)).dot(imagePoint)
    tempMat2 = np.linalg.inv(rvecM1).dot(tvec1)
    s = zConst + tempMat2[2]
    s /= tempMat[2]
    s = 1
    print("测试角点的深度值:", s)

    # 3D到2D转换cameraMatrix1、rvecM1, tvec1
    worldPoints = np.array([0, 0, 0, 1], dtype=np.float64)
    worldPoints = np.array([*worldBoxPoints[index], 1], dtype=np.float64)  # 假设imagePoint已经定义并赋值
    RT_ = np.hstack((rvecM1, tvec1))
    image_points = cameraMatrix1 @ RT_ @ worldPoints
    D_Points = image_points / image_points[2]
    print("测试角点对应的3D to 2D:", D_Points[:2])


    # 2D到3D转换s、cameraMatrix1、rvecM1, tvec1
    wcPoint = np.linalg.inv(rvecM1) @ (np.linalg.inv(cameraMatrix1) * s * imagePoint - tvec1)
    worldPoint = (wcPoint[0, 0], wcPoint[1, 0], wcPoint[2, 0])
    print("测试角点2D to 3D:", worldPoint)


    # # 原来其他角点
    # for point in boxPoints:
    #     # cv2.circle(sourceImage, tuple(point), 3, (0, 255, 0), -1)
    #     cv2.circle(sourceImage, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)
    #
    # cv2.imshow("Source", sourceImage)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # 创建3D图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 将世界坐标点组织为3D平面的顶点
    verts = [worldBoxPoints.astype(int)]

    # 绘制平面
    poly = Poly3DCollection(verts, alpha=0.5)
    ax.add_collection3d(poly)

    # 为每个3D世界坐标点添加标签
    for point in worldBoxPoints.astype(int):
        label = f'({point[0]}, {point[1]}, {point[2]})'
        ax.text(point[0], point[1], point[2], label)

    # 通过2D到3D反投影计算得到的3D点
    worldPoint_int = tuple(map(int, worldPoint))
    ax.scatter(*worldPoint_int, c='r', marker='x')

    # 将z坐标设置为零的3D点
    projected_point_int = (worldPoint_int[0], worldPoint_int[1], 0)
    ax.scatter(*projected_point_int, c='b', marker='o')

    plt.show()


if __name__ == "__main__":
    main()