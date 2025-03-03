﻿#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#pragma warning(disable : 4996)

//define constant
#define PI 3.1416
#define CHANNEL 3
#define FILTER_H 3
#define FILTER_W 3
#define WINDOW_SIZE 5
#define DEBUG
#define THRESHOLD 1000000000
float CONT_k = 0.23;

//struct definition
struct pixel {
    float H;
    float W;
    float R;
    float G;
    float B;
    float edge;
    float phase;
    float magnitude;
}typedef pixel;

using namespace cv;

//gradient filter definition
int filterX[9] = {
    -1, -1, -1,
    0,  0,  0,
    1,  1,  1
};
int filterY[9] = {
    -1, 0, 1,
    -1, 0, 1,
    -1, 0, 1
};

void gaussian(Mat *inoutImg, int sigma, int sizeFilter){
    int i, j, h, w;
    int filterHalf = (int)(sizeFilter / 2);
    int height = inoutImg->rows;
    int width = inoutImg->cols;
    float * gaussianFilter = (float *)calloc(sizeFilter*sizeFilter, sizeof(float));
    float sum=0;
    Mat tmp = Mat::zeros(height, width, CV_8UC3);
    pixel * temp = (pixel *)calloc(height*width, sizeof(pixel));
    float mediate = 0;
    for (h = 0; h < sizeFilter; h++) {
        for (w = 0; w < sizeFilter; w++) {
            mediate = pow(h - filterHalf, 2) + pow(w - filterHalf, 2);
            sum += gaussianFilter[h * sizeFilter + w] = (1 / (pow(sigma, 2) * 2 * PI)) * exp((-1) * (mediate/(2*sigma*sigma)));
            std::cout << gaussianFilter[h * sizeFilter + w] << std::endl;
        }
    }
    std::cout << sum<< std::endl;
    sum = 1/sum;
    for (i = 0; i < height; i++) {
        //X
        for (j = 0; j < width; j++) {
            //convolution calculation
            for (h = 0; h < sizeFilter; h++) {
                for (w = 0; w < sizeFilter; w++) {
                    //countinue when outbound
                    if ((i - filterHalf + h < 0) || (i - filterHalf + h >= height)) continue;
                    if ((j - filterHalf + w < 0) || (j - filterHalf + w >= width)) continue;
                    //calc MAC gradient filter
                    tmp.at<Vec3b>(i* width + j)[0] += (float)(inoutImg->at<Vec3b>(i - filterHalf + h, j - filterHalf + w)[0]) * sum * (gaussianFilter[h * sizeFilter + w]);
                    tmp.at<Vec3b>(i* width + j)[1] += (float)(inoutImg->at<Vec3b>(i - filterHalf + h, j - filterHalf + w)[1]) * sum * (gaussianFilter[h * sizeFilter + w]);
                    tmp.at<Vec3b>(i* width + j)[2] += (float)(inoutImg->at<Vec3b>(i - filterHalf + h, j - filterHalf + w)[2]) * sum * (gaussianFilter[h * sizeFilter + w]);
                    
                }
            }        
            //std::cout << i << j << "  ";
        }
    }
    for (i = 0; i < height; i++) {
        //X
        for (j = 0; j < width; j++) {
            inoutImg->at<Vec3b>(i, j)[0] = tmp.at<Vec3b>(i * width + j)[0];
            inoutImg->at<Vec3b>(i, j)[1] = tmp.at<Vec3b>(i * width + j)[1];
            inoutImg->at<Vec3b>(i, j)[2] = tmp.at<Vec3b>(i * width + j)[2];
        }
    }
}

int main() {
    //open img
    Mat origImg = imread("photo2.jpg", cv::IMREAD_COLOR);
    int height = origImg.rows;
    int width = origImg.cols;

    //define iteration variable
    int i, j, h, w;

    Mat origImgGray = Mat::zeros(height, width, CV_8UC1);
    gaussian(&origImg, 3, 3);
    //conv. color to gray
    for (i = 0; i < height; i++) {
        //X
        for (j = 0; j < width; j++) {
            origImgGray.at<uchar>(i, j) = (origImg.at<Vec3b>(i, j)[0] + origImg.at<Vec3b>(i, j)[1] + origImg.at<Vec3b>(i, j)[2]) / 3;
        }
    }

    Mat outputImg(height, width, CV_8UC1);

    //allocation output pixel type
    pixel* output = (pixel*)calloc(height * width, sizeof(pixel));

    int max = INT_MIN+1, min = INT_MAX;

    float comp;
    for (i = 0; i < height; i++) {
        //X
        for (j = 0; j < width; j++) {
            //convolution calculation
            for (h = 0; h < FILTER_H; h++) {
                for (w = 0; w < FILTER_W; w++) {
                    //countinue when outbound
                    if ((i - 1 + h < 0) || (i - 1 + h >= height)) continue;
                    if ((j - 1 + w < 0) || (j - 1 + w >= width)) continue;
                    //calc MAC gradient filter
                    output[i * width + j].W += (origImgGray.at<uchar>(i - 1 + h, j - 1 + w)) * (filterX[h * FILTER_W + w]);
                    output[i * width + j].H += (origImgGray.at<uchar>(i - 1 + h, j - 1 + w)) * (filterY[h * FILTER_W + w]);
                }
            }
        }
    }
    float det=0, tr=0;
    float MatA=0, MatB=0, MatC=0, MatD=0;
    int convHalf = (int)(WINDOW_SIZE / 2);
    for (i = 0; i < height; i++) {
        //X
        for (j = 0; j < width; j++) {
            //convolution calculation
            for (h = 0; h < WINDOW_SIZE; h++) {
                for (w = 0; w < WINDOW_SIZE; w++) {
                    //countinue when outbound
                    if ((i - convHalf + h < 0) || (i - convHalf + h >= height)) continue;
                    if ((j - convHalf + w < 0) || (j - convHalf + w >= width)) continue;
                    //calc MAC gradient filter
                    MatA += pow(output[(i - convHalf + h) * width + j - convHalf + w].W, 2);
                    MatB += output[(i - convHalf + h) * width + j - convHalf + w].W * output[(i - convHalf + h) * width + j - convHalf + w].H;
                    MatD += pow(output[(i - convHalf + h) * width + j - convHalf + w].H, 2);
                    //output[i * width + j].W += (origImgGray.at<uchar>(i - (WINDOW_SIZE / 2) + h, j - (WINDOW_SIZE / 2) + w)) * (filterX[h * WINDOW_SIZE + w]);
                    //output[i * width + j].H += (origImgGray.at<uchar>(i - (WINDOW_SIZE / 2) + h, j - (WINDOW_SIZE / 2) + w)) * (filterY[h * WINDOW_SIZE + w]);
                }
            }
            det = MatA * MatD - pow(MatB,2);
            tr = (MatA + MatD)* (MatA + MatD)* CONT_k;
            
            output[i * width + j].edge = det - tr;
            det = tr = 0;
            MatA = MatB = MatC = MatD = 0;
            max = MAX((int)output[i * width + j].edge, max);
        }
    }
    
    Scalar c;
    Point pCenter;
    int radius = 3;
    
    
#ifdef DEBUG
    printf("%d", max);
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            if (THRESHOLD < output[i * width + j].edge) {
                pCenter.x = j;
                pCenter.y = i;
                c.val[0] = 0;
                c.val[1] = 0;
                c.val[2] = 255;
                circle(origImg, pCenter, radius, c, 2, 8, 0);
                //std::cout << output[i * width + j].edge;
                //std::cout << std::endl;
            }
        }
        //;
    }
    //show result for debug
    //imwrite("output.bmp", outputImg);
    imshow("orig", origImg);
    imwrite("output.bmp", origImg);
    //imshow("result", outputImg);
    waitKey(5000);
#endif // DEBUG
}