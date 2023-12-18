import cv2
import numpy as np
import xml.etree.ElementTree as ET
#
def load_parameters():
    with np.load(r'F:\920yijia3072\920yijia3072.npz') as data:
        mat_intri = data['mat_intri']
        coff_dis = data['coff_dis']
    return mat_intri, coff_dis



def load_calibration_from_xml():
    file_path = 'D:\Desktop\Calibration-ZhangZhengyou-Method-master\Fhaikang2.npz'
    tree = ET.parse(file_path)
    root = tree.getroot()

    # 修改此处
    mat_intri = np.array([float(x.text) for x in root.find('camera_matrix').findall('data')]).reshape(3, 3)
    coff_dis = np.array([float(x.text) for x in root.find('camera_distortion').findall('data')])

    return mat_intri, coff_dis


# 读取图像
sourceImage = cv2.imread(r"F:\920yijia3072\gaokong\5.jpg")
gray = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)

# 检测棋盘格角点
pattern_size = (8, 6)
found, corners = cv2.findChessboardCorners(gray, pattern_size)
if not found:
    print("无法找到棋盘格角点，请检查图像和棋盘格大小。")  # 无法找到棋盘格角点
else:
    # 选择图像中的8个角点，例如第一行的第一个和最后一个，第二行的第一个和最后一个，以此类推
    boxPoints = np.array(
        [
            corners[0][0], corners[pattern_size[0] - 1][0],
            corners[pattern_size[0]][0], corners[2 * pattern_size[0] - 1][0],
            corners[-2 * pattern_size[0]][0], corners[-pattern_size[0] - 1][0],
            corners[-pattern_size[0]][0], corners[-1][0]
        ],
        dtype=np.float32)

    # 在图像上标识这8个角点，并显示顺序
    for i, corner in enumerate(boxPoints):
        x, y = tuple(corner)
        cv2.circle(sourceImage, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(sourceImage, str(i + 1), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    print("角点坐标:\n", boxPoints)



    # 定义8个角点的世界坐标
    worldBoxPoints = np.array(
        [
            [0, 0, 0], [0.14, 0, 0],  # 第一行
            [0, 0.02, 0], [0.14, 0.02, 0],  # 第二行
            [0, 0.10, 0], [0.14, 0.10, 0],  # 倒数第二行
            [0, 0.12, 0], [0.14, 0.12, 0]  # 最后一行
        ],
        dtype=np.float32)

    print("世界坐标:\n", worldBoxPoints)


    # 加载相机内参和畸变系数
    cameraMatrix1, distCoeffs1 = load_parameters()
    # cameraMatrix1, distCoeffs1 = load_calibration_from_xml()
    print("相机内参:\n", cameraMatrix1)
    print("畸变系数:\n", distCoeffs1)

    ret, rvecs, tvecs, _ = cv2.solvePnPGeneric(worldBoxPoints, boxPoints, cameraMatrix1, distCoeffs1)
    rvec = rvecs[0]
    tvec = tvecs[0]






    #保存旋转向量和平移向量
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
    index = 7

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



    # 生成随机像素点
    height, width = sourceImage.shape[:2]
    random_points = np.random.randint([0, 0], [height, width], size=(20, 2))

    # 映射随机像素点到3D世界坐标
    for i, random_point in enumerate(random_points):
        uvPoint = np.array([random_point[0], random_point[1], 1])
        tempMat = np.linalg.inv(np.dot(cameraMatrix1, R) - uvPoint.reshape(-1, 1) @ tvec.reshape(1, -1))
        s = Zconst + np.dot(tempMat[-1, :], tvec)
        s /= np.dot(tempMat[-1, :], uvPoint)
        worldCoord = np.dot(R.T, (np.linalg.inv(cameraMatrix1) @ (s * uvPoint) - tvec)).flatten()
        print(f"随机像素点 {i + 1} 的世界坐标: x={worldCoord[0]}, y={worldCoord[1]}, z={worldCoord[2]}")

        # 在图像上标识这些随机像素点
        x, y = random_point
        cv2.circle(sourceImage, (int(x), int(y)), 5, (255, 0, 0), -1)  # 画圆标记像素点
        cv2.putText(sourceImage, str(i + 1), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # 标记顺序


    # 缩放图像
    frame =sourceImage

    import cv2 as cv
    scale_percent = 50  # 缩放到原来的200%
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)

    cv.imshow("frame", resized_frame)
    cv.waitKey(0)
    cv.destroyAllWindows()
