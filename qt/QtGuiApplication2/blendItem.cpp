#include "blendItem.h"



void BlendItem::setBg(cv::Mat _bgImg) {
    backGround = _bgImg;
    bgHeight = backGround.rows;
    bgWidth = backGround.cols;
}

void BlendItem::blendItem(cv::Mat _alphaImg, cv::Point location) {
    alphaImg = _alphaImg;
    blendCenterW = location.x;
    blendCenterH = location.y;
    alphaHeight = alphaImg.rows;
    alphaWidth = alphaImg.cols;
    for (int alphaH = 0; alphaH < alphaHeight; alphaH++) {
        for (int alphaW = 0; alphaW < alphaWidth; alphaW++) {

            //calc blend location of BG IMG
            bgH = blendCenterH + alphaH - alphaHeight / 2;
            bgW = blendCenterW + alphaW - alphaWidth / 2;
            if (bgH > 0 && bgW > 0 && bgH < bgHeight-1 && bgW < bgWidth-1) {

                //alpha velue
                alphaV = alphaImg.at<Vec4b>(alphaH, alphaW)[3];
                //blend pixel
                backGround.at<Vec3b>(bgH, bgW)[0] = (alphaV * (float)alphaImg.at<Vec4b>(alphaH, alphaW)[0] + (255 - alphaV) * (float)backGround.at<Vec3b>(bgH, bgW)[0])/255;
                backGround.at<Vec3b>(bgH, bgW)[1] = (alphaV * (float)alphaImg.at<Vec4b>(alphaH, alphaW)[1] + (255 - alphaV) * (float)backGround.at<Vec3b>(bgH, bgW)[1])/255;
                backGround.at<Vec3b>(bgH, bgW)[2] = (alphaV * (float)alphaImg.at<Vec4b>(alphaH, alphaW)[2] + (255 - alphaV) * (float)backGround.at<Vec3b>(bgH, bgW)[2])/255;

            }
            //imshow("1", backGround);
            //waitKey(1);

        }
    }
}

void BlendItem::getBlended(cv::Mat &output) {
    backGround.copyTo(output);
}