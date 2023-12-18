#include <opencv2\opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;

int main()
{
    // 读取图片
    vector<Mat> images;
    string imageName;
    ifstream fin("calibdata.ini");
    while (getline(fin, imageName))
    {
        Mat img = imread(imageName);
        if (!img.empty())
        {
            images.push_back(img);
        }
    }

    // 获取图像大小
    Size imageSize = images[0].size();
    Size board_size = Size(7, 5);
    vector<vector<Point2f>> cornersPixel;

    // 寻找并提取棋盘角点
    for (int i = 0; i < images.size(); i++)
    {
        Mat gray;
        cvtColor(images[i], gray, COLOR_BGR2GRAY);
        vector<Point2f> img_points;
        if (findChessboardCorners(gray, board_size, img_points, 2))
        {
            find4QuadCornerSubpix(gray, img_points, Size(5, 5));
            cornersPixel.push_back(img_points);
        }
    }

    // 设置棋盘格子大小
    Size squareSize = Size(5, 5);
    vector<vector<Point3f>> cornersSpace;

    // 计算真实的点坐标
    for (int i = 0; i < images.size(); i++)
    {
        vector<Point3f> realPointSet;
        for (int j = 0; j < board_size.height; j++)
        {
            for (int k = 0; k < board_size.width; k++)
            {
                Point3f realPoint;
                realPoint.x = k * squareSize.width;
                realPoint.y = j * squareSize.height;
                realPoint.z = 0;
                realPointSet.push_back(realPoint);
            }
        }
        cornersSpace.push_back(realPointSet);
    }

    // 初始化相机参数
    Mat cameraMatrix = Mat::zeros(3, 3, CV_32FC1);
    Mat distCoeffs = Mat::zeros(1, 5, CV_32FC1);
    vector<Mat> rvecs, tvecs;

    // 相机标定
    calibrateCamera(cornersSpace, cornersPixel, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, 0);

    vector<Mat> RRvecs, tvecss;

    // 使用RANSAC算法求解旋转和平移矩阵
    for (int i = 0; i < images.size(); i++)
    {
        Mat rvecRansac, tvecRansac;
        solvePnPRansac(cornersSpace[i], cornersPixel[i], cameraMatrix, distCoeffs, rvecRansac, tvecRansac, 0);
        Mat RRansac;
        Rodrigues(rvecRansac, RRansac);
        RRvecs.push_back(RRansac);
        tvecss.push_back(tvecRansac);
    }

    // 计算投影矩阵
    Mat outcan, outcan2, cam;
    Mat zeri = (Mat_<double>(4, 1) << 0, 0, 0, 1);
    Mat zerii = (Mat_<double>(1, 4) << 0, 0, 0, 1);
    Mat zero = (Mat_<double>(3, 1) << 0, 0, 0);
    hconcat(RRvecs[0], tvecss[0], outcan);
    Mat  Fin = cameraMatrix * outcan;
    Mat Inv;
    invert(Fin, Inv, DECOMP_SVD);

    // 计算相机矩阵
    hconcat(cameraMatrix, zero, cam);
    vconcat(outcan, zerii, outcan2);
    Mat sa = cam * outcan2 * zeri;
    double s =  sa.at<double>(2,0);
    Mat a = sa / s;

    // 输入像素坐标
    int  pixel_x, pixel_y;
    cin >> pixel_x >> pixel_y;
    double pixel_xx = 320 - pixel_x + a.at<double>(0,0);
    double pixel_yy = 240 - pixel_y + a.at<double>(1,0);
    Mat proPixel = (Mat_<double>(3, 1) << pixel_xx, pixel_yy, 1);

    // 计算真实坐标
    Mat_<double> pmacc = 163.84  * s* Inv * proPixel;

    // 输出需要移动的距离
    cout << "需要移动的距离：1j^" << pmacc.at<double>(0) <<",2j^"<< pmacc.at<double>(1);

    return 0;
}
