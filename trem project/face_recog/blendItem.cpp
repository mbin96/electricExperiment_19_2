#include "blendItem.h"

void BlendItem::setInput(cv::Mat _bgImg, cv::Mat _alphaImg, cv::Point location) {
    backGround = _bgImg;
    alphaImg = _alphaImg;
    blendCenterW = location.x;
    blendCenterH = location.y;
    alphaHeight = alphaImg.rows;
    alphaWidth = alphaImg.cols;
    bgHeight = backGround.rows;
    bgWidth = backGround.cols;
}

void BlendItem::getBlended(cv::Mat output) {
    for (int alphaH = 0; alphaH < alphaHeight; alphaH++) {
        for (int alphaW = 0; alphaW < alphaWidth; alphaW++) {

            //calc blend location of BG IMG
            bgH = blendCenterH + alphaH - alphaHeight / 2;
            bgW = blendCenterW + alphaW - alphaWidth / 2;
            if (bgH > 0 && bgW > 0 && bgH < bgHeight && bgW < bgWidth) {

                //alpha velue
                alphaV = alphaImg.at<Vec4b>(alphaH, alphaW)[3] / 255;
                //blend pixel
                backGround.at<Vec3b>(bgH, bgW)[0] = alphaV * alphaImg.at<Vec4b>(alphaH, alphaW)[0] + (1 - alphaV) * backGround.at<Vec3b>(bgH, bgW)[0];
                backGround.at<Vec3b>(bgH, bgW)[1] = alphaV * alphaImg.at<Vec4b>(alphaH, alphaW)[1] + (1 - alphaV) * backGround.at<Vec3b>(bgH, bgW)[1];
                backGround.at<Vec3b>(bgH, bgW)[2] = alphaV * alphaImg.at<Vec4b>(alphaH, alphaW)[2] + (1 - alphaV) * backGround.at<Vec3b>(bgH, bgW)[2];

            }
            //imshow("1", backGround);
            //waitKey(1);

        }
    }
    output = backGround;
}