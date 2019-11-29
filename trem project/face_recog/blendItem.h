#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
using namespace cv;
class BlendItem {

private:
    cv::Mat backGround;
    cv::Mat alphaImg;
    int alphaHeight;
    int alphaWidth;
    int bgHeight;
    int bgWidth;
    int blendCenterW;
    int blendCenterH;
    float alphaV;
    int bgH, bgW;

public:
    void setInput(cv::Mat _bgImg, cv::Mat _alphaImg, cv::Point location);
    void getBlended(cv::Mat output);

};
