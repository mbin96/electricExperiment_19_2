#include <vector>
#include <math.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"





class TrackImg {

private:
    cv::Point kemonoEarPoint;
    cv::Vec3d eav;
    cv::Mat current_shape;
    cv::Point noseTop;
    int numLandmarks;
    //int itemFaceWidth;
    float faceWidth;

public:
    bool isFace = false;
    //input face Img
    cv::Mat cameraImg;
    void setKemonoEar(cv::Mat& _inOutAlphaImg, cv::Point& kemonoLocation, int AlphaImgFaceWidth);
    void inputLandmark(int numLandmarks, cv::Mat &current_shape, cv::Vec3d eav);
};

void TrackImg::inputLandmark(int _numLandmarks, cv::Mat &_current_shape, cv::Vec3d eav) {
    numLandmarks = _numLandmarks;
    current_shape = _current_shape;

}

void TrackImg::setKemonoEar(cv::Mat& _inOutAlphaImg, cv::Point& kemonoLocation, int AlphaImgFaceWidth) {

    cv::Point facebrowLeft = cv::Point(current_shape.at<float>(17), current_shape.at<float>(17 + numLandmarks));
    cv::Point facebrowRight = cv::Point(current_shape.at<float>(26), current_shape.at<float>(26 + numLandmarks));
    cv::Point faceEyeCenter = cv::Point((current_shape.at<float>(21) + current_shape.at<float>(22)) / 2, (current_shape.at<float>(21 + numLandmarks) + current_shape.at<float>(22 + numLandmarks)) / 2);
    cv::Point faceBottom = cv::Point(current_shape.at<float>(8), current_shape.at<float>(8 + numLandmarks));

    noseTop = cv::Point(current_shape.at<float>(30), current_shape.at<float>(30 + numLandmarks));
    int faceHigh = sqrt(pow(faceEyeCenter.x - noseTop.x, 2) + pow(faceEyeCenter.y - noseTop.y, 2));
    if (faceHigh < 25) {
        current_shape = cv::Mat::zeros(1, 1, CV_8UC1);
    }
    int faceCenterX = current_shape.at<float>(2) / 2 + current_shape.at<float>(14) / 2;
    // y �����ִ°� �󸶳� �������� �����غ���
    cv::Point faceTop = faceEyeCenter - cv::Point(0.7 * (faceBottom.x - faceCenterX), faceHigh * pow(cos(eav[2] * 3.14 / 180), 2));
    //faceTop.x = 2 * faceCenter.x - faceBottom.x;
    faceWidth = sqrt(pow(facebrowLeft.x - facebrowRight.x, 2) + pow(facebrowLeft.y - facebrowRight.y, 2));
    //faceTop.y = faceCenter.y - faceWidth / ((faceBottom.y - current_shape.at<float>(30 + numLandmarks))/ (faceBottom.y - faceCenter.y));
    cv::resize(_inOutAlphaImg, _inOutAlphaImg, cv::Size(_inOutAlphaImg.cols * (faceWidth / AlphaImgFaceWidth), _inOutAlphaImg.rows * (0.8 * sqrt(pow(faceBottom.x - faceEyeCenter.x, 2) + pow(faceBottom.y - faceEyeCenter.y, 2))) / AlphaImgFaceWidth));//(cos(eav[0]*3.14 / 180))));
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point(_inOutAlphaImg.rows / 2, _inOutAlphaImg.cols / 2), -eav[2], 1.0);
    cv::warpAffine(_inOutAlphaImg, _inOutAlphaImg, rot, cv::Size(_inOutAlphaImg.cols, _inOutAlphaImg.rows), 1, cv::BORDER_TRANSPARENT);

    /*
    for (int j = 0; j < numLandmarks; j++) {
        int x = current_shape.at<float>(j);
        int y = current_shape.at<float>(j + numLandmarks);
        std::stringstream ss;
        ss << j;
        cv::putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.3, cv::Scalar(0, 0, 255));

        //Mat outImg = cv::Mat::zeros(LBP_INPUT_SIZE, LBP_INPUT_SIZE, CV_8UC1);

        //lbpImg(lbpCut(Image, x, y)).copyTo(outImg);

        //cv::imshow("lbp" + std::to_string(j), lbpImg(lbpCut(Image, x, y)));


        //            cv::putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 0, 255));
        cv::circle(Image, cv::Point(x, y), 10, cv::Scalar(0, 0, 255), -1);
    }
    */
    kemonoLocation = faceTop;
}




/*

int learning() {

    cv::VideoCapture mCamera("face.mp4");
    //cv::VideoCapture mCamera(0);

    if (!mCamera.isOpened()) {
        std::cout << "Camera opening failed..." << std::endl;
        system("pause");
        return 0;
    }
    BlendItem blend;

    //png ������ alpha ä�� �״�� �ҷ����� 
    Mat kemonoEar = imread("rabbit_ear_1.png", cv::IMREAD_UNCHANGED);
    Mat frame;
    Mat alphaItem, image;
    TrackImg trackFace;
    Point itemPoint;
    while (1) {
        mCamera >> frame;
        frame.copyTo(image);
        kemonoEar.copyTo(alphaItem);
        trackFace.loadLandmark();
        if (trackFace.setLandmark(image)) {
            trackFace.setKemonoEar(alphaItem, itemPoint, 180);
            blend.setBg(image);
            blend.blendItem(alphaItem, itemPoint);
            blend.getBlended(image);
        }
    }

    return 0;
}

*/