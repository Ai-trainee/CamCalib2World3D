

# Cam2World3D

## Project Overview
Cam2World3D provides a suite of tools for camera calibration and converting 2D image coordinates to 3D world coordinates. This project is suitable for various applications requiring spatial recognition and measurement, such as augmented reality and machine vision.

## Directory Structure
- `config/` - Stores configuration files for project settings and parameter configuration.
- `pic/` - Sample images for documentation or testing.
- `utils/` - Utility scripts to assist with project-related tasks.

## Main Files
- `calibrate_helper.py` - Utility script to assist with camera calibration.
- `run_calib_IR.py` - Script for RGB camera calibration that also performs image undistortion (includes an additional step of applying correction coefficients compared to `run_calib_RGB.py`).
- `run_calib_RGB.py` - Script for RGB camera calibration.
- `Cam2World3D` - This module implements the 3D conversion from image coordinates to world coordinates. The main implementation methods include PnP-based and plane-line-based algorithms. There are many more details inside that you can explore. The latest implementation is `stackpnp+s100+8点.py`.

## Installation Guide
Description of the installation process:
```bash
git clone https://github.com/Ai-trainee/CamCalib2World3D.git
cd CamCalib2World3D
```

## Usage Instructions
Provides basic usage instructions on how to run the scripts for calibration (refer to the above for coordinate conversion).
```bash
# Run RGB camera calibration
python run_calib_RGB.py
# Run RGB camera calibration with undistortion
python run_calib_IR.py
```
## Demo
![Snipaste_2023-08-28_16-07-45.png](doc%2FSnipaste_2023-08-28_16-07-45.png)
![Snipaste_2023-08-28_16-10-03.png](doc%2FSnipaste_2023-08-28_16-10-03.png)
![Snipaste_2023-08-28_16-50-48.png](doc%2FSnipaste_2023-08-28_16-50-48.png)
![Snipaste_2023-09-01_17-30-18.png](doc%2FSnipaste_2023-09-01_17-30-18.png)
## How to Contribute
We welcome contributions in all forms, whether it be suggestions for new features, code fixes, or documentation updates.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
If you have any questions or suggestions, please contact us via:
- Email: [xx邮箱]
- GitHub Issue: [链接到仓库的 Issues 页面]
