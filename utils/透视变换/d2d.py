# 导入所需库
import cv2
import numpy as np

def main():
    # 读取图像
    sourceImage = cv2.imread(r"1.jpg")


    boxPoints = np.array(
        [
            [799,222], [1036,195],  # 第一行
            [537,1024], [1228,992]  # 最后一行
        ],
        dtype=np.float32)

    print("角点坐标:\n", boxPoints)

    # 定义8个角点的世界坐标
    worldBoxPoints = np.array(
        [
            [0, 0, 0], [2, 0, 0],  # 第一行
            [0, 8.4, 0], [2, 8.4, 0]  # 最后一行
        ],
        dtype=np.float32)

    print("世界坐标:\n", worldBoxPoints)

    # 计算单应性矩阵
    h, status = cv2.findHomography(boxPoints, worldBoxPoints)

    # # 使用单应性矩阵将图像坐标转换为世界坐标
    # index = 2
    # imagePoint = np.array([*boxPoints[index], 1], dtype=np.float64)
    # worldPoint = h @ imagePoint
    # worldPoint = worldPoint / worldPoint[2]
    #
    # # 将结果四舍五入为整数
    # worldPoint_int = np.round(worldPoint[:2]).astype(int)
    # print("测试角点2D to 2D (整数):", worldPoint_int)


    # 假设像素点坐标
    pixel_point = np.array([717, 462, 1], dtype=np.float64)

    # 使用之前计算的单应性矩阵h将像素点转换为世界坐标
    world_point = h @ pixel_point
    world_point = world_point / world_point[2]

    # 输出世界坐标
    print("自定义点的世界坐标:", world_point[:2])


# 主函数调用
if __name__ == "__main__":
    main()
