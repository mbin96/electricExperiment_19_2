#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <fstream>
#include <Windows.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/ml/ml.hpp>
#include "ldmarkmodel.h"
#include "lbpFunction.hpp"
#include "main.hpp"


#include "blendItem.h"

using namespace std;
using namespace cv;


int learning() {
    //CascadeClassifier cascadeFace;
    //cascadeFace.load("haarcascade_frontalface_alt.xml");
    //std::ofstream out("LBP.txt");
    
    
    
    ldmarkmodel modelt;
    std::string modelFilePath = "roboman-landmark-model.bin";
    while (!load_ldmarkmodel(modelFilePath, modelt)) {
        std::cout << "zzzz." << std::endl;
        std::cin >> modelFilePath;
    }
    
    
    cv::VideoCapture mCamera("face.mp4");
    //cv::VideoCapture mCamera(0);
           
    if (!mCamera.isOpened()) {
        std::cout << "Camera opening failed..." << std::endl;
        system("pause");
        return 0;
    }
    //png 파일을 alpha 채널 그대로 불러오기 
    Mat kemonoEar = imread("cat_ear.png", cv::IMREAD_UNCHANGED);
    cv::Mat blendItem;
    cv::Mat Image;
    cv::Mat current_shape;
    //아이템의 얼굴 너비가 얼마인지 알아야 정확한 얼굴길이와 매칭이 가능
    int itemFaceWidth=160;
    Point faceBottom, faceCenter, faceTop, faceLeft, faceRight;
    float faceWidth, faceHeight;
    Mat origImg;
    vector<Rect> faces;
    
    
    
    BlendItem blend;
    while (1) {
        mCamera >> origImg;
        resize(origImg, origImg, Size(1280, 720));

        //TODO - input from map
        //Mat origImg = imread("photo.jpg", cv::IMREAD_COLOR);
        
        //imshow("debug", origImg);
        //waitKey(5000);
           
        

        origImg.copyTo(Image);
        kemonoEar.copyTo(blendItem);
        //Image = kemonoEar + Image;
        //alphaBlend(Image, kemonoEar, Point(300, 300));

        modelt.track(Image, current_shape);
        cv::Vec3d eav;
        modelt.EstimateHeadPose(current_shape, eav);
        modelt.drawPose(Image, current_shape, 50);

        //눈썹 사이 - 이미지 블랜드 되는 중심점 21 22 사이
        //눈썹 가장 먼 거리 - 이미지 에서 얼굴길이로 지정한 가로길이
        // 눈썹 중심에서 9번(턱)까지 길이 이미지 세로길이
        //눈썹 17 - 21, 22 - 26
        //코끝 30
        int numLandmarks = current_shape.cols / 2;
        if (numLandmarks > 0) {
            Point facebrowLeft = Point(current_shape.at<float>(17), current_shape.at<float>(17 + numLandmarks));
            Point facebrowRight = Point(current_shape.at<float>(26), current_shape.at<float>(26 + numLandmarks));
            Point faceEyeCenter = Point((current_shape.at<float>(21) + current_shape.at<float>(22)) / 2, (current_shape.at<float>(21 + numLandmarks) + current_shape.at<float>(22 + numLandmarks)) / 2);
            faceBottom = Point(current_shape.at<float>(8),current_shape.at<float>(8 + numLandmarks));
            faceCenter.x = current_shape.at<float>(2)/2 + current_shape.at<float>(14)/2;
            faceCenter.y = current_shape.at<float>(2 + numLandmarks) / 2 + current_shape.at<float>(14 + numLandmarks) / 2;
            faceLeft = Point(current_shape.at<float>(1), current_shape.at<float>(1 + numLandmarks));
            faceRight = Point(current_shape.at<float>(15), current_shape.at<float>(15 + numLandmarks));
            
            int faceHigh = sqrt(pow(faceEyeCenter.x - current_shape.at<float>(30), 2) + pow(faceEyeCenter.y - current_shape.at<float>(30 + numLandmarks), 2));

            // y 더해주는거 얼마나 더해줄지 생각해보기
            faceTop = faceEyeCenter - Point(-faceCenter.x+faceBottom.x, faceHigh);
            //faceTop.x = 2 * faceCenter.x - faceBottom.x;
            faceWidth = sqrt(pow(facebrowLeft.x - facebrowRight.x,2) + pow(facebrowLeft.y - facebrowRight.y,2));
            //faceTop.y = faceCenter.y - faceWidth / ((faceBottom.y - current_shape.at<float>(30 + numLandmarks))/ (faceBottom.y - faceCenter.y));
            resize(blendItem, blendItem, Size(blendItem.rows * (faceWidth / itemFaceWidth), blendItem.rows * (cos(eav[0]*3.14 / 180))));
            Mat rot = getRotationMatrix2D(Point(blendItem.rows / 2, blendItem.cols / 2), -eav[2], 1.0);
            
            warpAffine(blendItem, blendItem, rot,Size(blendItem.cols, blendItem.rows),1,BORDER_TRANSPARENT);
            imshow("a", blendItem);
            blend.setInput(Image, blendItem, faceTop);
            blend.getBlended(Image);
        }
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
            cv::circle(Image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
        }

        cv::imshow("Camera", Image);
        
        waitKey(20);
    }

    system("pause");
    return 0;
}




//getFaceImg
//return imgSize*imgSize of Mat array
//must use delete[] array of unallocation
Mat * getFaceImg(Mat input, int imgSize, vector<Rect> faces) {
    
    Mat * faceDetected = new Mat[faces.size()];

    for (int y = 0; y < faces.size(); y++) {
        int X, Y, H, W;
        X = faces[y].x - 30;
        Y = faces[y].y - 30;
        H = faces[y].y + faces[y].height + 30;
        W = faces[y].x + faces[y].width + 30;
        
        //Mat faceDetected[y] = input(rect) //is it possible?
        faceDetected[y] = Mat::zeros(H - Y, W - X, CV_8UC3);
        
        //crop to faceDetected[y] from input
        
        for (int h = Y; h < H; h++) {
            for (int w = X; w < W; w++) {
                if (h > 0 && w > 0 && h < input.rows && w < input.cols) {
                    faceDetected[y].at<Vec3b>(h - Y, w - X)[0] = (input.at<Vec3b>(h, w)[0]);//B
                    faceDetected[y].at<Vec3b>(h - Y, w - X)[1] = (input.at<Vec3b>(h, w)[1]);//G
                    faceDetected[y].at<Vec3b>(h - Y, w - X)[2] = (input.at<Vec3b>(h, w)[2]);//R
                }
            }
        }

        resize(faceDetected[y], faceDetected[y], Size(imgSize, imgSize));
    }
    return faceDetected;
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
