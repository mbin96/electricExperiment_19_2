#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/ml/ml.hpp>
#include "ldmarkmodel.h"
#include "lbpFunction.hpp"
#include "main.hpp"

using namespace std;
using namespace cv;


int learning() {

    std::ofstream out("LBP.txt");

    //TODO - input from map
    Mat origImg = imread("photo.jpg", cv::IMREAD_COLOR);
    vector<Rect> faces = doCascacade(origImg);
    Mat * testimgarray=getFaceImg(origImg, FACE_IMG_SIZE, faces);
    origImg = testimgarray[0];

    //imshow("debug", origImg);
    //waitKey(5000);
    
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
    //modelt.EstimateHeadPose(current_shape, eav);
    //modelt.drawPose(Image, current_shape, 50);
    
    int numLandmarks = current_shape.cols / 2;

    //ÅÎ ±×¸®±â
    //for (int j = 0; j < 16; j++) {
    //    int x = current_shape.at<float>(j);
    //    int y = current_shape.at<float>(j + numLandmarks);
    //    std::stringstream ss;
    //    ss << j;
    //    //            cv::putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 0, 255));
    //    cv::circle(Image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
    //}

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


//do cascacade and return vector fo face square
vector<Rect> doCascacade(Mat input) {
    CascadeClassifier cascadeFace;
    cascadeFace.load("haarcascade_frontalface_alt.xml");
    vector<Rect> faces;
    cascadeFace.detectMultiScale(input, faces, 1.1, 4, 0 | 2, Size(10, 10));
    return faces;
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
                faceDetected[y].at<Vec3b>(h - Y, w - X)[0] = (input.at<Vec3b>(h, w)[0]);//B
                faceDetected[y].at<Vec3b>(h - Y, w - X)[1] = (input.at<Vec3b>(h, w)[1]);//G
                faceDetected[y].at<Vec3b>(h - Y, w - X)[2] = (input.at<Vec3b>(h, w)[2]);//R
            }
        }

        resize(faceDetected[y], faceDetected[y], Size(imgSize, imgSize));
    }
    return faceDetected;
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
