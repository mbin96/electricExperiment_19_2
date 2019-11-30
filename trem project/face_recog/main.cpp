#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <fstream>
#include <Windows.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/objdetect/objdetect.hpp"
//#include <opencv2/ml/ml.hpp>
//#include "ldmarkmodel.h"
//#include "lbpFunction.hpp"
#define MAIN
//#include "trackImg.h"
#include "blendItem.h"
//#include "ldmarkmodel.h"


using namespace std;
using namespace cv;


class TrackImg {

public:
    void loadLandmark();
    void setKemonoEar(cv::Mat _inOutAlphaImg, cv::Point kemonoLocation, int AlphaImgFaceWidth);
    bool setLandmark(cv::Mat input);
};



int learning() {
    
    cv::VideoCapture mCamera("face.mp4");
    //cv::VideoCapture mCamera(0);

    if (!mCamera.isOpened()) {
        std::cout << "Camera opening failed..." << std::endl;
        system("pause");
        return 0;
    }
    BlendItem blend;

    //png 파일을 alpha 채널 그대로 불러오기 
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
        cv::imshow("Camera", image);
        waitKey(20);
    }


    system("pause");
    return 0;
}




int main()
{
    //POINT p;
    //GetCursorPos(&p);
   // double mouseX = p.x;
    //double mouseY = p.y;
   // SetCursorPos(0, 0);

    int sel;
    //cout << "mode sel >";
    //cin >> sel;

    //if (sel == 0) {
    learning();
    //}
    return 0;
}
