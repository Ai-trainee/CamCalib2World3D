import cv2
import numpy as np
def load_parameters():
    with np.load('./25.npz') as data:
        mat_intri = data['mat_intri']
        coff_dis = data['coff_dis']
    return mat_intri, coff_dis

def main():
    # 读取图像
    # sourceImage = cv2.imread(r"D:\Desktop\Calibration-ZhangZhengyou-Method-master\bd\IMG_20230809_183605.jpg")
    sourceImage = cv2.imread(r"D:\Desktop\Calibration-ZhangZhengyou-Method-master\temp\suofang\112.jpg")
    # sourceImage = resize_image(sourceImage) # 调整图像大小
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


    # 定义世界坐标，只包括XY平面坐标
    worldBoxPoints = np.array([[0, 0], [20, 0], [20, 14], [0, 14]], dtype=np.float32)
    #打印世界坐标
    print("世界坐标:\n",worldBoxPoints)


    #相机内参矩阵 与 畸变系数
    cameraMatrix1, distCoeffs1 = load_parameters()

    # 使用PnP求解R&T时，将世界坐标增加一个维度，添加Z轴坐标为0
    worldBoxPoints3D = np.hstack([worldBoxPoints, np.zeros((4, 1), dtype=np.float32)])
    _, rvec1, tvec1, inliers = cv2.solvePnPRansac(worldBoxPoints3D, boxPoints, cameraMatrix1, distCoeffs1)

    rvecM1, _ = cv2.Rodrigues(rvec1)

    # 计算旋转角度
    thetaZ = np.arctan2(rvecM1[1, 0], rvecM1[0, 0]) / np.pi * 180
    thetaY = np.arctan2(-1 * rvecM1[2, 0], np.sqrt(rvecM1[2, 1]**2 + rvecM1[2, 2]**2)) / np.pi * 180
    thetaX = np.arctan2(rvecM1[2, 1], rvecM1[2, 2]) / np.pi * 180
    # print("theta x:", thetaX, "theta Y:", thetaY, "theta Z:", thetaZ)



    # 计算参数s (深度)
    index = 1  # 这里的序号是0，因为您正在使用boxPoints的第一个元素
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
    worldPoints = np.array([*worldBoxPoints3D[index], 1], dtype=np.float64)  # 使用3D世界坐标
    RT_ = np.hstack((rvecM1, tvec1))
    image_points = cameraMatrix1 @ RT_ @ worldPoints
    D_Points = image_points / image_points[2]
    print("测试角点对应的3D to 2D:", D_Points[:2])


    # 2D到3D转换s、cameraMatrix1、rvecM1, tvec1
    wcPoint = np.linalg.inv(rvecM1) @ (np.linalg.inv(cameraMatrix1) * s * imagePoint - tvec1)
    worldPoint = (wcPoint[0, 0], wcPoint[1, 0], wcPoint[2, 0])
    print("测试角点2D to 3D:", worldPoint)





    # 原来其他角点
    for point in boxPoints:
        # cv2.circle(sourceImage, tuple(point), 3, (0, 255, 0), -1)
        cv2.circle(sourceImage, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)

    cv2.imshow("Source", sourceImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
