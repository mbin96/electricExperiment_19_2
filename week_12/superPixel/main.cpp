#include "SLIC.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#pragma warning(disable : 4996)
using namespace cv;

Mat getSlic(Mat img, int m_spcount = 200, double m_compactness = 20) {
    
    uint* imgARGB = (uint*)calloc(img.cols * img.rows, sizeof(int));

    //Mat(BGR) to ARGB
    for (int h = 0; h < img.cols; h++) {
        for (int w = 0; w < img.rows; w++) {
            imgARGB[h * img.rows + w] = (0xff << 24) +
                (((uint)img.at<cv::Vec3b>(h, w)[2]) << 16) +
                (((uint)img.at<cv::Vec3b>(h, w)[1]) << 8) +
                ((uint)img.at<cv::Vec3b>(h, w)[0]);
        }
    }

    int width = img.rows;
    int height = img.cols;

    int sz = width * height;
    //if (m_spcount > sz);
    int* labels = new int[sz];
    int numlabels(0);
    SLIC slic;
    slic.PerformSLICO_ForGivenK(imgARGB, width, height, labels, numlabels, m_spcount, m_compactness);//for a given number K of superpixels
    //slic.PerformSLICO_ForGivenStepSize(img, width, height, labels, numlabels, m_stepsize, m_compactness);//for a given grid step size
    //slic.DrawContoursAroundSegments(img, labels, width, height, 0);//for black contours around superpixels
    slic.DrawContoursAroundSegmentsTwoColors(imgARGB, labels, width, height);//for black-and-white contours around superpixels

    //slic.SaveSuperpixelLabels(labels, width, height, picvec[k], saveLocation);
    if (labels) delete[] labels;

    //picHand.SavePicture(img, width, height, picvec[k], saveLocation, 1, "_SLICO");// 0 is for BMP and 1 for JPEG)

    for (int h = 0; h < img.cols; h++) {
        for (int w = 0; w < img.rows; w++) {
            img.at<cv::Vec3b>(h, w)[2] = 0xff & (imgARGB[h * img.rows + w] >> 16);
            img.at<cv::Vec3b>(h, w)[1] = 0xff & (imgARGB[h * img.rows + w] >> 8);
            img.at<cv::Vec3b>(h, w)[0] = 0xff & (imgARGB[h * img.rows + w] >> 0);
        }
    }

    if (imgARGB) delete[] imgARGB;
    return img;
}

int main()
{
    int m_spcount = 200;//super pixel size
    double m_compactness = 20;
    



    CascadeClassifier cascadeFace;
    cascadeFace.load("C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml");

    VideoCapture capture(0);

    if (!capture.isOpened()) {
        printf("WTF");
        return 0;
    }

    Mat frameRef;

    vector<Rect> faces;
    while (true) {
        capture >> frameRef;
        cascadeFace.detectMultiScale(frameRef, faces, 1.1, 4, 0 | 2, Size(10, 10));
        Mat faceDetected;
        std::cout << faces.size() << std::endl;
        for (int y = 0; y < faces.size(); y++) {
            Point lb(faces[y].x + faces[y].width, faces[y].y + faces[y].height);
            Point tr(faces[y].x, faces[y].y);
            Rect faceRect(faces[y].x, faces[y].y, faces[y].x + faces[y].width, faces[y].y + faces[y].height);

            faceDetected = Mat::zeros(faces[y].height, faces[y].width, CV_8UC3);

            

            for (int h = faces[y].y; h < faces[y].y + faces[y].height; h++) {
                for (int w = faces[y].x; w < faces[y].x + faces[y].width; w++) {
                    faceDetected.at<Vec3b>(h - faces[y].y, w - faces[y].x)[0] = (frameRef.at<Vec3b>(h, w)[0]);
                    faceDetected.at<Vec3b>(h - faces[y].y, w - faces[y].x)[1] = (frameRef.at<Vec3b>(h, w)[1]);
                    faceDetected.at<Vec3b>(h - faces[y].y, w - faces[y].x)[2] = (frameRef.at<Vec3b>(h, w)[2]);
                }
            }

            Mat img = getSlic(faceDetected, m_spcount, m_compactness);
            for (int h = faces[y].y; h < faces[y].y + faces[y].height; h++) {
                for (int w = faces[y].x; w < faces[y].x + faces[y].width; w++) {
                    (frameRef.at<Vec3b>(h, w)[0]) = img.at<Vec3b>(h - faces[y].y, w - faces[y].x)[0];
                    (frameRef.at<Vec3b>(h, w)[1]) = img.at<Vec3b>(h - faces[y].y, w - faces[y].x)[1];
                    (frameRef.at<Vec3b>(h, w)[2]) = img.at<Vec3b>(h - faces[y].y, w - faces[y].x)[2];
                }
            }
            rectangle(frameRef, lb, tr, Scalar(0, 0, 255), 3, 8, 0);
        }

        imshow("img", frameRef);

        if (waitKey(30) >= 0) {

        }
    }




    waitKey(5000);
    printf("Done!");
}