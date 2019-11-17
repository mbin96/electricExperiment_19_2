#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "ldmarkmodel.h"
#include "lbpFunction.hpp"

using namespace std;
using namespace cv;


int learning() {

    std::ofstream out("LBP.txt");


    Mat origImg = imread("photo.jpg", cv::IMREAD_COLOR);


    ldmarkmodel modelt;
    std::string modelFilePath = "roboman-landmark-model.bin";
    while (!load_ldmarkmodel(modelFilePath, modelt)) {
        std::cout << "zzzz." << std::endl;
        std::cin >> modelFilePath;
    }
    /*
    cv::VideoCapture mCamera(0);
    if (!mCamera.isOpened()) {
        std::cout << "Camera opening failed..." << std::endl;
        system("pause");
        return 0;
    }
    */
    cv::Mat Image;
    cv::Mat current_shape;

    origImg.copyTo(Image);

    modelt.track(Image, current_shape);
    cv::Vec3d eav;
    modelt.EstimateHeadPose(current_shape, eav);
    modelt.drawPose(Image, current_shape, 50);

    int numLandmarks = current_shape.cols / 2;

    for (int j = 0; j < 16; j++) {
        int x = current_shape.at<float>(j);
        int y = current_shape.at<float>(j + numLandmarks);
        std::stringstream ss;
        ss << j;
        //            cv::putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 0, 255));
        cv::circle(Image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
    }

    for (int j = 16; j < numLandmarks; j++) {
        int x = current_shape.at<float>(j);
        int y = current_shape.at<float>(j + numLandmarks);
        std::stringstream ss;
        ss << j;
        cv::putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 0, 255));


        Mat outImg = cv::Mat::zeros(LBP_INPUT_SIZE, LBP_INPUT_SIZE, CV_8UC1);

        lbpImg(lbpCut(Image, x, y)).copyTo(outImg);

        cv::imshow("lbp" + std::to_string(j), lbpImg(lbpCut(Image, x, y)));

        for (int h = 0; h < LBP_INPUT_SIZE; h++) {
            for (int w = 0; w < LBP_INPUT_SIZE; w++) {
                out << (uint)outImg.at<uchar>(h, w) << endl;
            }
        }



        //            cv::putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 0, 255));
        cv::circle(Image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
    }

    cv::imshow("Camera", Image);
    waitKey(5000);


    system("pause");
    return 0;
}


int main()
{
    int sel;
    cout << "mode sel >";
    cin >> sel;

    if (sel == 0) {
        learning();
    }
    return 0;
}
