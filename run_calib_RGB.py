# -*- coding: utf-8 -*-
"""
Calibrate the Camera with Zhang Zhengyou Method.
Picture File Folder: "./pic/RGB_camera_calib_img/", Without Distortion.

By You Zhiyuan, 2022.07.04, zhiyuanyou@foxmail.com
"""

import os

from calibrate_helper import Calibrator
import cv2
import numpy as np
def main():
    img_dir = "./pic/RGB_camera_calib_img"
    img_dir = r"F:\920yijia3072\gaokong"
    shape_inner_corner = (8,6)
    # shape_inner_corner = (11,8)
    size_grid = 0.02
    # create calibrator
    calibrator = Calibrator(img_dir, shape_inner_corner, size_grid)
    # calibrate the camera
    mat_intri, coff_dis = calibrator.calibrate_camera()
    # 保存标定参数
    np.savez('F:\920yijia3072/gaokong/gaokong.npz', mat_intri=mat_intri, coff_dis=coff_dis)

    #转换成科学计数法
    np.set_printoptions(suppress=True)
    print("mat_intri:\n", mat_intri)
    print("coff_dis:\n", coff_dis)
    print("参数保存成功！")


if __name__ == '__main__':
    main()
