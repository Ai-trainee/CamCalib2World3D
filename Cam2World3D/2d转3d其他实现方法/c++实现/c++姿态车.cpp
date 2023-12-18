#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

int main() {
    // 通过标定板的图像像素坐标和对应的世界坐标，使用PnP求解相机的R&T（旋转和平移矩阵）//
    Point2f point;
    vector<Point2f> boxPoints; // 存储像素坐标

    // 加载图像
    Mat sourceImage = imread("2.bmp");
    namedWindow("Source", 1);

    // 设置标定板在图像中的角点
    // 第一个点
    point = Point2f((float)558, (float)259); // 640X480
    boxPoints.push_back(point);
    circle(sourceImage, boxPoints[0], 3, Scalar(0, 255, 0), -1, 8);

    // 第二个点
    point = Point2f((float)629, (float)212); // 640X480
    boxPoints.push_back(point);
    circle(sourceImage, boxPoints[1], 3, Scalar(0, 255, 0), -1, 8);

    // 第三个点
    point = Point2f((float)744, (float)260); // 640X480
    boxPoints.push_back(point);
    circle(sourceImage, boxPoints[2], 3, Scalar(0, 255, 0), -1, 8);

    // 第四个点
    point = Point2f((float)693, (float)209); // 640X480
    boxPoints.push_back(point);
    circle(sourceImage, boxPoints[3], 3, Scalar(0, 255, 0), -1, 8);

    // 设置标定板在现实世界中的角点
    vector<Point3f> worldBoxPoints; // 存入世界坐标
    Point3f tmpPoint;
    tmpPoint = Point3f((float)2750, (float)890, (float)0);
    worldBoxPoints.push_back(tmpPoint);
    tmpPoint = Point3f((float)3500, (float)450, (float)0);
    worldBoxPoints.push_back(tmpPoint);
    tmpPoint = Point3f((float)2790, (float)-240, (float)0);
    worldBoxPoints.push_back(tmpPoint);
    tmpPoint = Point3f((float)3620, (float)-50, (float)0);
    worldBoxPoints.push_back(tmpPoint);

    // 设置相机内参矩阵
    cv::Mat cameraMatrix1 = Mat::eye(3, 3, cv::DataType<double>::type);
    cv::Mat distCoeffs1(5, 1, cv::DataType<double>::type); // 畸变参数
    distCoeffs1.at<double>(0, 0) = 0.061439051;
    distCoeffs1.at<double>(1, 0) = 0.03187556;
    distCoeffs1.at<double>(2, 0) = -0.00726151;
    distCoeffs1.at<double>(3, 0) = -0.00111799;
    distCoeffs1.at<double>(4, 0) = -0.00678974;

    // 取自Mastring OpenCV的焦距和主点坐标
    double fx = 328.61652824;
    double fy = 328.56512516;
    double cx = 629.80551148;
    double cy = 340.5442837;
    cameraMatrix1.at<double>(0, 0) = fx;
    cameraMatrix1.at<double>(1, 1) = fy;
    cameraMatrix1.at<double>(0, 2) = cx;
    cameraMatrix1.at<double>(1, 2) = cy;

    // 使用PnP求解R&T（旋转和平移矩阵）
    cv::Mat rvec1(3, 1, cv::DataType<double>::type); // 旋转向量
    cv::Mat tvec1(3, 1, cv::DataType<double>::type); // 平移向量
    cv::solvePnP(worldBoxPoints, boxPoints, cameraMatrix1, distCoeffs1, rvec1, tvec1, false, CV_ITERATIVE);
    cv::Mat rvecM1(3, 3, cv::DataType<double>::type); // 旋转矩阵
    cv::Rodrigues(rvec1, rvecM1);

    // 此处用于求相机位于坐标系内的旋转角度,2D-3D转换并不用求
    // 计算旋转角度
    const double PI = 3.1415926;
    double thetaZ = atan2(rvecM1.at<double>(1, 0), rvecM1.at<double>(0, 0)) / PI * 180;
    double thetaY = atan2(-1 * rvecM1.at<double>(2, 0), sqrt(rvecM1.at<double>(2, 1) * rvecM1.at<double>(2, 1)
        + rvecM1.at<double>(2, 2) * rvecM1.at<double>(2, 2))) / PI * 180;
    double thetaX = atan2(rvecM1.at<double>(2, 1), rvecM1.at<double>(2, 2)) / PI * 180;
    cout << "theta x  " << thetaX << endl << "theta Y: " << thetaY << endl << "theta Z: " << thetaZ << endl;

    // 根据公式求Zc，即s
    cv::Mat imagePoint = cv::Mat::ones(3, 1, cv::DataType<double>::type);
    cv::Mat tempMat, tempMat2;
    // 输入一个2D坐标点，便可以求出相应的s
    imagePoint.at<double>(0, 0) = 558;
    imagePoint.at<double>(1, 0) = 259;
    double zConst = 0; // 实际坐标系的距离
    // 计算参数s
    double s;
    tempMat = rvecM1.inv() * cameraMatrix1.inv() * imagePoint;
    tempMat2 = rvecM1.inv() * tvec1;
    s = zConst + tempMat2.at<double>(2, 0);
    s /= tempMat.at<double>(2, 0);
    cout << "s : " << s << endl;

    // 3D到2D的转换
    cv::Mat worldPoints = Mat::ones(4, 1, cv::DataType<double>::type);
    worldPoints.at<double>(0, 0) = 3620;
    worldPoints.at<double>(1, 0) = -590;
    worldPoints.at<double>(2, 0) = 0;
    cout << "world Points :  " << worldPoints << endl;
    Mat image_points = Mat::ones(3, 1, cv::DataType<double>::type);
    Mat RT_;
    hconcat(rvecM1, tvec1, RT_);
    cout << "RT_" << RT_ << endl;
    image_points = cameraMatrix1 * RT_ * worldPoints;
    Mat D_Points = Mat::ones(3, 1, cv::DataType<double>::type);
    D_Points.at<double>(0, 0) = image_points.at<double>(0, 0) / image_points.at<double>(2, 0);
    D_Points.at<double>(1, 0) = image_points.at<double>(1, 0) / image_points.at<double>(2, 0);
    cout << "3D to 2D:   " << D_Points << endl;

    // 相机坐标
    Mat camera_coordinates = -rvecM1.inv() * tvec1;

    // 2D到3D的转换
    cv::Mat imagePoint_your_know = cv::Mat::ones(3, 1, cv::DataType<double>::type); // u,v,1
    imagePoint_your_know.at<double>(0, 0) = 558;
    imagePoint_your_know.at<double>(1, 0) = 259;
    Mat wcPoint = rvecM1.inv() * (cameraMatrix1.inv() * s * imagePoint_your_know - tvec1);
    Point3f worldPoint(wcPoint.at<double>(0, 0), wcPoint.at<double>(1, 0), wcPoint.at<double>(2, 0));
    cout << "2D to 3D :" << worldPoint << endl;

    imshow("Source", sourceImage);
    waitKey(0);
    return 0;
}

// 封装函数，用于实现图像坐标到世界坐标的转换
Point3f getWorldPoints(Point2f inPoints) {
    cv::Mat imagePoint = cv::Mat::ones(3, 1, cv::DataType<double>::type); // u,v,1
    imagePoint.at<double>(0, 0) = inPoints.x;
    imagePoint.at<double>(1, 0) = inPoints.y;
    Mat wcPoint = rotationMatrix.inv() * (s * cameraMatrix.inv() * imagePoint - tvec);
    Point3f worldPoint(wcPoint.at<double>(0, 0), wcPoint.at<double>(1, 0), wcPoint.at<double>(2, 0));
    return worldPoint;
}

