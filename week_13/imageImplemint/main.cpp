#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;
#define BOUND 30
int main(int argc, char** argv)
{
    // read image and error pattern
    //Mat img = imread("rena.bmp");

    cv::VideoCapture mCamera("movie2.mp4");
    if (!mCamera.isOpened()) {
        std::cout << "Camera opening failed..." << std::endl;
        system("pause");
        return 0;
    }
    Mat img;
    while (1) {
        mCamera >> img;
        resize(img, img, Size(640, 360));
        Mat img_gray;
        cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
        Mat reconstructed;
        Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
        Mat contImg;

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        blur(img_gray, img_gray, Size(3, 3));
        Canny(img_gray, contImg, 100, 100 * 3, 3);
        //imshow("contImg", contImg);
        findContours(contImg, contours, hierarchy, 3, 2, Point(0, 0));
        for (int i = 0; i < contours.size(); i++)
        {
            //Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            drawContours(img_gray, contours, i, 255, 1, 8, hierarchy, 0, Point());
        }

        for (int h = 0; h < img_gray.rows; h += 3) {
            for (int w = 0; w < img_gray.cols; w += 3) {
                int count = 0;
                for (int hh = 0; hh < BOUND; hh++) {
                    for (int ww = 0; ww < BOUND; ww++) {
                        int height = h + hh - (BOUND / 2);
                        int width = w + ww - (BOUND / 2);
                        if ((height < img_gray.rows) && (width < img_gray.cols) && (height > 0) && (width > 0))
                           if(img_gray.at<uchar>(height, width) == 255)
                            count ++;
                    }
                }
                if (count > 150) {
                    for (int hh = 0; hh < BOUND; hh++) {
                        for (int ww = 0; ww < BOUND; ww++) {
                            int height = h + hh - (BOUND / 2);
                            int width = w + ww - (BOUND / 2);
                            if ((height < img.rows) && (width < img.cols) && (height > 0) && (width > 0))
                                mask.at<uchar>(height, width) = 255;
                              
                        }
                    }
                }
                
                    
            }
        }
        imshow("mask", mask);
        inpaint(img, mask, reconstructed, 3, cv::INPAINT_TELEA);
        imshow("reconstructed image", reconstructed);
        imshow("orignal image", img);
        waitKey(10);
    }
    return 0;
}