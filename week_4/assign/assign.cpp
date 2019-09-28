#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <iostream>
#include <stdlib.h>

//header
#include "visionFunctions.hpp"


int main() {
    //open img
    int height, width;

    //calc vision weight of ref Img.
    Mat refImg = imread("ref.bmp", cv::IMREAD_COLOR);
    height = refImg.rows;
    width = refImg.cols;
    pixel * visionWeightRef = setVision(refImg);

    //calc vision weight of compare Img.
    Mat compImg = imread("tarShift_1.bmp", cv::IMREAD_COLOR);
    height = compImg.rows;
    width = compImg.cols;
    pixel * visionWeightComp = setVision(compImg);

    findSameEdge(refImg, compImg, visionWeightRef, visionWeightComp);
    waitKey(5000);
}



