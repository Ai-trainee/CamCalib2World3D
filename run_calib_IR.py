# -*- coding: utf-8 -*-
"""
Calibrate the Camera with Zhang Zhengyou Method.
Picture File Folder: "./pic/IR_camera_calib_img/", With Distortion. 

By You Zhiyuan, 2022.07.04, zhiyuanyou@foxmail.com
"""

import os

import numpy as np

from calibrate_helper import Calibrator


def main():
    img_dir = "./pic/IR_camera_calib_img"
    # img_dir ="./老旧suofang"
    img_dir = r"F:\picss4\picss640"
    img_dir = r"F:\sxt3_1280x720"
    shape_inner_corner = (11, 8)
    shape_inner_corner = (8, 6)
    size_grid = 0.02
    # create calibrator
    calibrator = Calibrator(img_dir, shape_inner_corner, size_grid)
    # calibrate the camera
    mat_intri, coff_dis = calibrator.calibrate_camera()
    # dedistort and save the dedistortion result
    save_dir = "./pic/IR_dedistortion_suofang"
    save_dir = r"F:\picss4\picss640\IR_dedistortion"
    save_dir = r"sxt3_1280x720\IR"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    calibrator.dedistortion(save_dir)
    # 保存标定参数
    # np.savez('./姿态估计pnp/老旧suofang.npz', mat_intri=mat_intri, coff_dis=coff_dis)
    np.savez(r'图像坐标转3d/now/sxt3_1280x720.npz', mat_intri=mat_intri, coff_dis=coff_dis)
if __name__ == '__main__':
    main()
