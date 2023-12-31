

# Cam2World3D

## 项目简介
Cam2World3D 提供了一套工具，用于相机标定和将2D图像坐标转换为3D世界坐标。该项目适用于需要进行空间识别和测量的各类应用，例如增强现实、机器视觉等。

## 目录结构
- `config/` - 存放配置文件，用于项目设置和参数配置。
- `pic/` - 示例图片，用于说明或测试。
- `utils/` - 实用工具脚本，辅助进行项目相关任务。

## 主要文件
- `calibrate_helper.py` - 辅助相机标定的工具脚本。
- `run_calib_IR.py` - 执行RGB相机标定的脚本执行图像矫正（比run_calib_RGB.py多了使用矫正系数这一步）
- `run_calib_RGB.py` - 执行RGB相机标定的脚本。
- `Cam2World3D` - 这个模块下实现了图像坐标到世界坐标的3d转换,主要实现方式有基于pnp的，还有基于平面直线的算法,具体里面还有很多，可以点进去查看详情,最新的stackpnp+s100+8点.py

## 安装指南
描述安装过程：
```bash
git clone https://github.com/Ai-trainee/CamCalib2World3D.git
cd CamCalib2World3D
```

## 使用说明
提供一个基本的使用说明，如何运行脚本进行标定(坐标转换参照上述)。
```bash
# 运行RGB相机标定
python run_calib_RGB.py
# 运行RGB相机标定后去畸变
python run_calib_IR.py
```
## 演示
![Snipaste_2023-08-28_16-07-45.png](doc%2FSnipaste_2023-08-28_16-07-45.png)
![Snipaste_2023-08-28_16-10-03.png](doc%2FSnipaste_2023-08-28_16-10-03.png)
![Snipaste_2023-08-28_16-50-48.png](doc%2FSnipaste_2023-08-28_16-50-48.png)
![Snipaste_2023-09-01_17-30-18.png](doc%2FSnipaste_2023-09-01_17-30-18.png)
## 如何贡献
我们欢迎所有形式的贡献，无论是新功能的建议、代码修正还是文档更新。

## 许可证
该项目采用 MIT 许可证 - 有关详细信息，请查看 LICENSE 文件。

## 联系方式
如果您有任何问题或建议，请通过以下方式联系我们：
- 邮箱：[xx邮箱]
- GitHub Issue：[链接到仓库的 Issues 页面]


