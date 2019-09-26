#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <iostream>
#include <stdlib.h>
//def header
#include "visionFunction.h"

using namespace cv;

void harris(pixel* output, int height, int width);
void gaussian(cv::Mat* inoutImg, int sigma, int sizeFilter);
void pointCircling(Mat* inoutImg, int y, int x, int sizeRadius, uchar b, uchar g, uchar r);
void calcGredient(Mat* inputImg, pixel* output);


void hog(pixel *output, int y, int x, int tilesize, int height, int width) {
    int tileW = 17, tileH = 17, blkI, blkJ, sumBlk = 0;
    tileW = tileH = tilesize;
    int i, j, h, w;
    int startX = x - (int)tileW/2, startY = y - (int)tileH/2;
            
    for (h = 0; h < tileH; h++) {
        for (w = 0; w < tileW; w++) {
            //countinue when outbound
            if ((h + startY < 0) || (h + startY>= height)) continue;
            if ((w + startX < 0) || (w + startX >= width)) continue;
            //calc hog weight
            output[y * width + x].hog[((int)output[(h + startY) * width + (w + startX)].phase)] +=
                output[(h + startY) * width + (w + startX)].magnitude;
        }
    }

    //sum of all block pixel's magnitude
    for (int k = 0; k < 9; k++) {
        sumBlk += pow(output[y * width + x].hog[k], 2) + 0.000001;
    }

    //L-2 normalize weight
    for (int k = 0; k < 9; k++) {
        output[y * width + x].hog[k] = output[y * width + x].hog[k] / sqrt(sumBlk);
    }
}

//vision main function
void setVision(Mat origImg, pixel * output ) {
    int height = origImg.rows;
    int width = origImg.cols;

    //define iteration variable
    int i, j, h, w;

    Mat origImgGray = Mat::zeros(height, width, CV_8UC1);

    //low pass filter
    gaussian(&origImg, 1, 3);

    //conv. color to gray
    for (i = 0; i < height; i++) {
        //X
        for (j = 0; j < width; j++) {
            origImgGray.at<uchar>(i, j) = (origImg.at<Vec3b>(i, j)[0] + origImg.at<Vec3b>(i, j)[1] + origImg.at<Vec3b>(i, j)[2]) / 3;
        }
    }

    //allocation output pixel type
   
    //gradiant
    calcGredient(&origImgGray, output);

    //harris edge detect
    harris(output, height, width);

    //get edge's HOG weight
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            if (output[i * width + j].edge > 0) {
                hog(output, i, j, 17, height, width);
            }
        }
    }

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            if (output[i * width + j].edge > 0) {
                pointCircling(&origImg, i, j, 3, 0, 255, 255);
            }
        }
    }
}

void findSameEdge(Mat refImg, Mat compImg, pixel * visionRef, pixel * visionComp) {
    Mat concatImg;
    int height = refImg.rows;
    int width = refImg.cols;
    int min = INT_MAX;
    //define iteration variable
    int refi, refj, compi, compj, k;
    int compx, compy;
    float subWeight=0;
    hconcat(refImg, compImg, concatImg);

    for (refi = 0; refi < height; refi++) {
        for (refj = 0; refj < width; refj++) {
            if (visionRef[refi * width + refj].edge > 0) {
                for (compi = 0; compi < height; compi++) {
                    for (compj = 0; compj < width; compj++) {
                        if (visionComp[compi * width + compj].edge > 0) {
                            for (k = 0; k < 9; k++) {
                                subWeight += fabs(visionRef[refi * width + refj].hog[k] - 
                                    visionComp[compi * width + compj].hog[k]);
                            }
                            if (min > subWeight) {
                                min = subWeight;
                                compx = compj;
                                compy = compi;
                            }
                            subWeight = 0;
                        }
                        
                    }
                }
                line(concatImg,Point(refj, refi),Point(compx+width,compy),Scalar(255,0,0),2,9,0);
                
                min = INT_MAX;
            }
        }
    }
    //line(concatImg, Point(52, 52), Point(264, 23), Scalar(255, 0, 0), 2, 9, 0);
    
    imshow("concat",concatImg);
    waitKey(5000);
}

int main() {
	//open img
    int height, width;

    //calc vision weight of ref Img.
	Mat refImg = imread("ref.bmp", cv::IMREAD_COLOR);
	height = refImg.rows;
	width = refImg.cols;
    pixel * visionRef = (pixel *)calloc(height * width, sizeof(pixel));
    setVision(refImg, visionRef);

    //calc vision weight of compare Img.
    Mat compImg = imread("ref.bmp", cv::IMREAD_COLOR);
    height = compImg.rows;
    width = compImg.cols;
    pixel * visionComp = (pixel*)calloc(height * width, sizeof(pixel));
    setVision(compImg, visionComp);

    findSameEdge(refImg, compImg, visionRef, visionComp);

	//show result
	imwrite("output.bmp", refImg);
	imshow("ref", refImg);
    imshow("comp", compImg);
	waitKey(5000);
}

//input grayscale Img and pixel struct's pointer(size must be same to input img)
//each pixel's gredient, quantized radian and normalized magnitude(0 to 255) will be save at pixel struct
void calcGredient(Mat* inputImg, pixel* output){
    int height = inputImg->rows;
    int width = inputImg->cols;

    int max = INT_MIN + 1, min = INT_MAX;
    int i, j, h, w;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {

            //convolution calculation
            for (h = 0; h < FILTER_H; h++) {
                for (w = 0; w < FILTER_W; w++) {
                    //countinue when outbound
                    if ((i - 1 + h < 0) || (i - 1 + h >= height)) continue;
                    if ((j - 1 + w < 0) || (j - 1 + w >= width)) continue;

                    //calc MAC gradient filter
                    output[i * width + j].W += (inputImg->at<uchar>(i - 1 + h, j - 1 + w)) * (filterX[h * FILTER_W + w]);
                    output[i * width + j].H += (inputImg->at<uchar>(i - 1 + h, j - 1 + w)) * (filterY[h * FILTER_W + w]);
                }
            }

            //conv radian to degree 0 ~ 179
            output[i * width + j].phase = 180 * (atan2(output[i * width + j].H, output[i * width + j].W) / PI);
            if (output[i * width + j].phase < 0) {
                output[i * width + j].phase = output[i * width + j].phase + 180;
            }
            if (output[i * width + j].phase >= 180) {
                output[i * width + j].phase = 0;
            }

            //Phase quantization by 20 degree(unsigned)
            output[i * width + j].phase = (int)(output[i * width + j].phase / 20);

            //calc magnitude
            output[i * width + j].magnitude = sqrt(pow(output[i * width + j].W, 2) + pow(output[i * width + j].H, 2));

            //set max, min for normalize
            max = __max(output[i * width + j].magnitude, max);
            min = __min(output[i * width + j].magnitude, min);
        }
    }

    //magnitude normalize
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            output[i * width + j].magnitude = 255 * (output[i * width + j].magnitude - min) / (max - min);
            //outputImg.at<uchar>(i, j) = output[i * width + j].magnitude;
        }
    }
}



//inputImg will be gaussian filtered
//size filter must odd number
void gaussian(cv::Mat* inoutImg, int sigma, int sizeFilter) {
    //iteration definition
    int i, j, h, w;

    int filterHalf = (int)(sizeFilter / 2);
    int height = inoutImg->rows;
    int width = inoutImg->cols;

    float* gaussianFilter = (float*)calloc(sizeFilter * sizeFilter, sizeof(float));
    float sum = 0;
    cv::Mat tmp = cv::Mat::zeros(height, width, CV_8UC3);

    for (h = 0; h < sizeFilter; h++) {
        for (w = 0; w < sizeFilter; w++) {
            gaussianFilter[h * sizeFilter + w] = pow(h - filterHalf, 2) + pow(w - filterHalf, 2);
            gaussianFilter[h * sizeFilter + w] = (1 / (pow(sigma, 2) * 2 * PI)) * exp((-1) * (gaussianFilter[h * sizeFilter + w] / (2 * sigma * sigma)));
            sum += gaussianFilter[h * sizeFilter + w];
            std::cout << gaussianFilter[h * sizeFilter + w] << std::endl;
        }
    }

    sum = 1 / sum;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            //convolution calculation
            for (h = 0; h < sizeFilter; h++) {
                for (w = 0; w < sizeFilter; w++) {
                    //countinue when outbound
                    if ((i - filterHalf + h < 0) || (i - filterHalf + h >= height)) continue;
                    if ((j - filterHalf + w < 0) || (j - filterHalf + w >= width)) continue;
                    //calc MAC gaussian filter and normalize
                    tmp.at<cv::Vec3b>(i * width + j)[0] += (float)(inoutImg->at<cv::Vec3b>(i - filterHalf + h, j - filterHalf + w)[0]) * sum * (gaussianFilter[h * sizeFilter + w]);
                    tmp.at<cv::Vec3b>(i * width + j)[1] += (float)(inoutImg->at<cv::Vec3b>(i - filterHalf + h, j - filterHalf + w)[1]) * sum * (gaussianFilter[h * sizeFilter + w]);
                    tmp.at<cv::Vec3b>(i * width + j)[2] += (float)(inoutImg->at<cv::Vec3b>(i - filterHalf + h, j - filterHalf + w)[2]) * sum * (gaussianFilter[h * sizeFilter + w]);
                }
            }
        }
    }

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            inoutImg->at<cv::Vec3b>(i, j)[0] = tmp.at<cv::Vec3b>(i * width + j)[0];
            inoutImg->at<cv::Vec3b>(i, j)[1] = tmp.at<cv::Vec3b>(i * width + j)[1];
            inoutImg->at<cv::Vec3b>(i, j)[2] = tmp.at<cv::Vec3b>(i * width + j)[2];
        }
    }
}

//input Img and x, y then thet pixel will be circled
void pointCircling(Mat* inoutImg, int y, int x, int sizeRadius, uchar b, uchar g, uchar r) {
    Scalar c;
    Point pCenter;
    pCenter.x = x;
    pCenter.y = y;
    c.val[0] = b;
    c.val[1] = g;
    c.val[2] = r;
    circle(*inoutImg, pCenter, sizeRadius, c, 2, 8, 0);
}

//input pixel struct then edge detect from it's gredient
void harris(pixel* output, int height, int width) {
    int i, j, h, w;
    float max = INT_MIN;
    float det = 0, tr = 0;
    float MatA = 0, MatB = 0, MatC = 0, MatD = 0;
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
                    //calc matrix
                    MatA += pow(output[(i - convHalf + h) * width + j - convHalf + w].W, 2);
                    MatB += output[(i - convHalf + h) * width + j - convHalf + w].W * output[(i - convHalf + h) * width + j - convHalf + w].H;
                    MatD += pow(output[(i - convHalf + h) * width + j - convHalf + w].H, 2);    
                }
            }
            det = MatA * MatD - pow(MatB, 2);
            tr = (MatA + MatD) * (MatA + MatD) * CONT_k;

            //raw data
            output[i * width + j].edge = det - tr;
            
            max = __max(max, (det - tr));
            //reset
            MatA = MatB = MatC = MatD = 0;
        }
    }
    for (i = 0; i < height; i++) {
        //X
        for (j = 0; j < width; j++) {
#ifdef DEBUG
            if (output[i * width + j].edge > 0)
                std::cout << output[i * width + j].edge ;
#endif // DEBUG
            //normalize -255 to 254
            output[i * width + j].edge = 1024 * (output[i * width + j].edge / max);
#ifdef DEBUG
            if (output[i * width + j].edge > 0)
                std::cout << output[i * width + j].edge<<std::endl;
#endif // DEBUG
            //edge weight to bool
            if (THRESHOLD < output[i * width + j].edge) {
                output[i * width + j].edge = 1;
            }else {
                output[i * width + j].edge = 0;
            }
        }
    }
    //std::cout << max << std::endl;
}

