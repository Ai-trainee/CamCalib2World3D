import cv2
import numpy as np

def load_parameters():
    with np.load(r'sxt.npz') as data:
        mat_intri = data['mat_intri']
        coff_dis = data['coff_dis']
    return mat_intri, coff_dis

# 读取图像
sourceImage = cv2.imread(r"/sxt_hk\Snipaste_2023-08-30_18-39-21.png")
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
    np.savez('RT.npz', rvec=rvec, tvec=tvec)
    print("RT.npz已保存")

    # 使用projectPoints投影世界坐标到像素坐标
    projected_points, _ = cv2.projectPoints(worldBoxPoints, rvec, tvec, cameraMatrix1, distCoeffs1)

    for i, point in enumerate(projected_points):
        print(f"角点 {i + 1} 投影像素坐标:", point[0])

    # 计算旋转矩阵
    R, _ = cv2.Rodrigues(rvec)

    # 定义射线方向（从相机中心到像素坐标）
    boxPoints_homogeneous = np.vstack([boxPoints.T, np.ones(boxPoints.shape[0])]) # 转换为齐次坐标
    ray_dirs = np.linalg.inv(cameraMatrix1).dot(boxPoints_homogeneous)

    # # 计算交点
    # for i, ray_dir in enumerate(ray_dirs.T):
    #     ray_dir = R.T.dot(ray_dir) # 转换到世界坐标系
    #     t = -tvec[2] / (R.T[2].dot(ray_dir)) # 计算射线与平面Z=0的交点
    #     world_point = t * ray_dir + R.T.dot(tvec).flatten() # 计算交点世界坐标
    #     print(f"角点 {i + 1} 世界坐标 (X, Y):", world_point[0], world_point[1])
    # 计算交点表面上为了改变符号
    #worldBoxPoints
    print("世界坐标:\n", worldBoxPoints)

    for i, ray_dir in enumerate(ray_dirs.T):
        ray_dir = R.T.dot(ray_dir)  # 转换到世界坐标系
        t = -tvec[2] / (R.T[2].dot(ray_dir))  # 计算射线与平面Z=0的交点
        world_point = -t * ray_dir - R.T.dot(tvec).flatten()  # 计算交点世界坐标，并更改符号
        print(f"角点 {i + 1} 世界坐标 (X, Y, Z): {format(world_point[0], '.2f')}, {format(world_point[1], '.2f')}, {format(world_point[2], '.2f')}")

        # 可视化圈出角点
        cv2.circle(sourceImage, (int(boxPoints[i][0]), int(boxPoints[i][1])), 5, (0, 255, 0), -1)


    import random

    # 随机选择一个像素点
    random_pixel = (random.randint(0, gray.shape[1] - 1), random.randint(0, gray.shape[0] - 1))

    # 转换为齐次坐标
    random_pixel_homogeneous = np.array([random_pixel[0], random_pixel[1], 1])

    # 定义射线方向（从相机中心到像素坐标）
    ray_dir = np.linalg.inv(cameraMatrix1).dot(random_pixel_homogeneous)

    # 转换到世界坐标系
    ray_dir = R.T.dot(ray_dir)

    # 计算射线与平面Z=0的交点
    t = -tvec[2] / (R.T[2].dot(ray_dir))

    # 计算交点世界坐标，并更改符号
    world_point = -t * ray_dir - R.T.dot(tvec).flatten()

    print(f"随机像素点 {random_pixel} 的世界坐标 (X, Y, Z):", world_point[0], world_point[1], world_point[2])

    # 获取图像的宽度和高度
    height, width = sourceImage.shape[:2]

    # 设置新的宽度和高度
    new_width = 800
    new_height = int((new_width / width) * height)

    # 使用cv2.resize调整图像大小
    resized_image = cv2.resize(sourceImage, (new_width, new_height))

    # 可视化圈出随机像素点
    cv2.circle(resized_image, (int(random_pixel[0] * new_width / width), int(random_pixel[1] * new_height / height)), 5,
               (0, 0, 255), -1)

    # 显示调整大小后的图像
    cv2.imshow('Random Pixel', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


