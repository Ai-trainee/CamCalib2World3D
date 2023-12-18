import cv2
import numpy as np

def load_parameters():
    with np.load(r'sxt3_1280x720.npz') as data:
        mat_intri = data['mat_intri']
        coff_dis = data['coff_dis']
    return mat_intri, coff_dis

# 读取图像
sourceImage = cv2.imread(r"frame_0.jpg")
gray = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)

# 检测棋盘格角点
pattern_size = (8, 6)
found, corners = cv2.findChessboardCorners(gray, pattern_size)
if not found:
    print("无法找到棋盘格角点，请检查图像和棋盘格大小。")
else:
    boxPoints = np.array(
        [corners[0][0], corners[pattern_size[0] - 1][0], corners[-pattern_size[0]][0], corners[-1][0]],
        dtype=np.float32)

    for i, corner in enumerate(boxPoints):
        x, y = tuple(corner)
        cv2.circle(sourceImage, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(sourceImage, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    worldBoxPoints = np.array([[0, 0, 0], [0.14, 0, 0], [0, 0.10, 0], [0.14, 0.10, 0]], dtype=np.float32)
    print("世界坐标:\n", worldBoxPoints)

    cameraMatrix1, distCoeffs1 = load_parameters()

    ret, rvec, tvec = cv2.solvePnP(worldBoxPoints, boxPoints, cameraMatrix1, distCoeffs1)

    R, _ = cv2.Rodrigues(rvec)

    projected_points, _ = cv2.projectPoints(worldBoxPoints, rvec, tvec, cameraMatrix1, distCoeffs1)

    for i, point in enumerate(projected_points):
        print(f"角点 {i + 1} 投影像素坐标:", point[0])

    # index = 3  # 选择一个角点的索引
    # pixel_point = boxPoints[index]
    # homogeneous_pixel = np.array([pixel_point[0], pixel_point[1], 1])
    #
    # # 使用第一个方法的数学公式
    # ray_dir = np.linalg.inv(cameraMatrix1).dot(homogeneous_pixel)
    # ray_dir = R.T.dot(ray_dir)
    # t = -tvec[2] / (R.T[2].dot(ray_dir))
    # world_point = -t * ray_dir - R.T.dot(tvec).flatten()
    #
    # X, Y, Z = world_point
    # print("世界坐标系下的角点坐标:", worldBoxPoints)
    # print(f"射线与地面交点的世界坐标: X={X}, Y={Y}, Z={Z}")
    #
    # cv2.circle(sourceImage, (int(pixel_point[0]), int(pixel_point[1])), 5, (255, 0, 0), -1)
    # cv2.putText(sourceImage, f"X={X:.2f}, Y={Y:.2f}", (int(pixel_point[0]), int(pixel_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    ## ...（其他代码未改动）

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







    # 生成随机像素点
    random_pixel = np.array([np.random.randint(0, gray.shape[1]), np.random.randint(0, gray.shape[0]), 1])

    # 计算射线方向（从相机中心到像素坐标）
    ray_dir_random = np.linalg.inv(cameraMatrix1).dot(random_pixel)

    # 转换到世界坐标系
    ray_dir_random = R.T.dot(ray_dir_random)

    # 计算射线与平面Z=0的交点
    t_random = -tvec[2] / (R.T[2].dot(ray_dir_random))

    # 计算交点世界坐标，并更改符号
    world_point_random = -t_random * ray_dir_random - R.T.dot(tvec).flatten()

    X_random, Y_random, Z_random = world_point_random
    print(f"随机像素点的世界坐标: X={X_random}, Y={Y_random}, Z={Z_random}")
    # 在原图像上标注随机像素点
    cv2.circle(sourceImage, (int(random_pixel[0]), int(random_pixel[1])), 5, (0, 0, 255), -1)
    cv2.putText(sourceImage, f"Random", (int(random_pixel[0]), int(random_pixel[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)




    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 创建3D图形对象
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 画出被映射的平面的四个世界坐标点
    worldBoxPoints = np.array([[0, 0, 0], [0.14, 0, 0], [0, 0.10, 0], [0.14, 0.10, 0]], dtype=np.float32)
    ax.scatter(worldBoxPoints[:, 0], worldBoxPoints[:, 1], worldBoxPoints[:, 2], c='b', marker='o',
               label='Plane Corners')
    # 定义平面的四个点
    plane_points = np.array([[0, 0, 0], [0.14, 0, 0], [0.14, 0.10, 0], [0, 0.10, 0]])

    # 创建平面的顶点坐标（X, Y, Z）
    x = plane_points[:, 0]
    y = plane_points[:, 1]
    z = plane_points[:, 2]

    # 绘制平面
    xx, yy = np.meshgrid(np.linspace(min(x), max(x), 50), np.linspace(min(y), max(y), 50))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='c', alpha=0.5)

    # 画出映射点
    ax.scatter(X, Y, Z, c='r', marker='^', label='Mapped Point')
    ax.scatter(X_random, Y_random, Z_random, c='g', marker='s', label='Random Mapped Point')

    # 设置坐标轴范围和标签
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 添加图例
    ax.legend()

    # 显示图形
    plt.show()

    # 缩放图像
    scale_percent = 100  # 缩放到原来的200%
    width = int(sourceImage.shape[1] * scale_percent / 100)
    height = int(sourceImage.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv2.resize(sourceImage, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('Image with Points', resized_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

