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
    pixel * frameWeightRef, * frameWeightTar;
    VideoCapture capture(0);
    /**/
    
    if (!capture.isOpened()) {
        printf("¾Æ¾Ç");
        return 0;
    }

    Mat frameRef;

    height = frameRef.rows;
    width = frameRef.cols;
    Mat frame;
    capture >> frameRef;
    frameWeightRef = setVision(frameRef, GAU_DISABLE, HOG_ENABLE);

    while (true) {
    
        capture >> frame;
        frameWeightTar = setVision(frame, GAU_DISABLE, HOG_ENABLE);
        imshow("video",findSameEdge(frameRef, frame, frameWeightRef, frameWeightTar));
        
        if (waitKey(1) >= 0) {
            capture >> frameRef;
            free(frameWeightRef);
            frameWeightRef = setVision(frameRef, GAU_DISABLE, HOG_ENABLE);
        }
        free(frameWeightTar);
    }

    /*
    //calc vision weight of ref Img.
    Mat refImg = imread("ref.bmp", cv::IMREAD_COLOR);
    height = refImg.rows;
    width = refImg.cols;
    pixel* visionWeightRef = setVision(refImg);

    //calc vision weight of compare Img.
    Mat compImg = imread("tarShift_1.bmp", cv::IMREAD_COLOR);
    height = compImg.rows;
    width = compImg.cols;
    pixel* visionWeightComp = setVision(compImg);

    findSameEdge(refImg, compImg, visionWeightRef, visionWeightComp);
    waitKey(5000);
    */
    
    }

