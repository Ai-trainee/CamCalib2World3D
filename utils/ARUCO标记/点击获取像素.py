import numpy as np
import cv2

img1 = cv2.imread(r'2023-09-19\192.168.1.40_01_20230919135632910.jpg')

# 像素点坐标初定义
pro_x = []
pro_y = []


# 定义鼠标点击事件并将点击坐标输入数组
def mouse_img_cod(event, cod_x, cod_y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (cod_x, cod_y)
        cv2.circle(img1, (cod_x, cod_y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img1, xy, (cod_x, cod_y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)  # 将坐标值放在图片内
        cv2.imshow("image", img1)
        pro_x.append(cod_x)
        pro_y.append(cod_y)


cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)  # 创建一个名为image的窗口
cv2.setMouseCallback("image", mouse_img_cod)  # 鼠标事件回调
cv2.imshow('image', img1)  # 显示图片
cv2.waitKey(0)  # 按下任意键退出窗口
cv2.destroyAllWindows()

print(pro_x[0], pro_y[0])  # 打印坐标值