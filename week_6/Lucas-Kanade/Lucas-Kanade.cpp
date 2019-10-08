#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/video/tracking.hpp>
#include <math.h>
#include <iostream>
#include <stdlib.h>
//header
#include "visionFunctions.hpp"



int main() {
    //open img
    int height, width;
    pixel * frameWeightPost, * frameWeightPre;
    VideoCapture capture(0);

    if (!capture.isOpened()) {
        printf("already opened");
        return 0;
    }
    
    Mat framePost;
    capture >> framePost;
    resize(framePost,framePost, Size(320, 240));
    height = framePost.rows;
    width = framePost.cols;
    Mat framePre;
    Mat frameOut;
        
    while (true) {

        capture >> framePre;
        resize(framePre, framePre, Size(320, 240));
        framePre.copyTo(frameOut);
        frameWeightPre = setVision(framePre, GAU_DISABLE, HOG_DISABLE);
        
        lucasKanade(frameOut, framePre, framePost, frameWeightPre);
        imshow("video", frameOut);

        framePre.copyTo(framePost);
        free(frameWeightPre);
        waitKey(10);

    }


}

