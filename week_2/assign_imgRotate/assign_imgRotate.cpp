#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <iostream>

#pragma warning(disable : 4996)

//define constant
#define PI 3.1416
#define CHANNEL 3
#define CONV_MAT_TO_CENTER 0
#define CONV_CENTOR_TO_MAT 1

struct pixel {
    float H;
    float W;
}typedef pixel;

using namespace cv;

void convCoordinate(pixel* tmp, pixel center, int conv);
void biLinear(Mat* outputImg, Mat origImg, float degree, float scale);

int main() {
    //open img
    Mat origImg = imread("hanazono.jpg", cv::IMREAD_COLOR);

    //input scale and degree
    float degree, scale;
    for (;;) {
        std::cout << "input scale > ";
        std::cin >> scale;
        if (scale > 0) {
            break;
        }
    }
    std::cout << "input rotate degree > ";
    std::cin >> degree;

    //conv. degree to radian
    degree = - PI * (degree / 180);
    
    //create new scaled outputImg Mat
    Mat outputImg = Mat::zeros(origImg.rows * scale, origImg.cols * scale, CV_8UC3);

    //calc. biLinear
    biLinear(&outputImg, origImg, degree, scale);
    
    //show result and save to bmp
    imshow("orig", origImg);
    imshow("result", outputImg);
    waitKey(5000);
    imwrite("output.bmp", outputImg);
}

void convCoordinate(pixel * tmp, pixel center, int conv){
    //conv. coordinate for rotation
    //rotation need centered 2d coordinate
    if (conv == CONV_MAT_TO_CENTER){
        tmp->W = tmp->W - center.W;
        tmp->H = center.H - tmp->H;
    }else if (conv == CONV_CENTOR_TO_MAT) {
        tmp->W = tmp->W + center.W;
        tmp->H = -tmp->H + center.H;
    }
}


void biLinear(Mat * outputImg, Mat origImg, float degree, float scale) {
    float biP1P3, biP1P2, biP3P4, biP2P4;
    float scaleInv = 1 / scale;
    pixel tmp, input, center;
    center.H = outputImg->rows / 2;
    center.W = outputImg->cols / 2;

    //calc like center is 0,0
    //Y
    for (int i = 0; i < outputImg->rows; i++) {
        //X
        for (int j = 0; j < outputImg->cols; j++) {
            
            tmp.W = j;
            tmp.H = i;
            //conv. coordinate and apply inverted rotation matrix
            convCoordinate(&tmp, center, CONV_MAT_TO_CENTER);
            input.W = tmp.W * cos(degree) + tmp.H * sin(degree);
            input.H = (-tmp.W) * sin(degree) + (tmp.H) * cos(degree);
            convCoordinate(&input, center, CONV_CENTOR_TO_MAT);
            
            //scale down for input H,W
            input.H = input.H * scaleInv;
            input.W = input.W * scaleInv;

            for (int k = 0; k < CHANNEL; k++) {
                if (input.H + 1 < origImg.rows && input.W + 1 < origImg.cols && input.H >= 0 && input.W >= 0) {
                    //calc horize and portrait weight
                    float weightH = input.H - (int)input.H;
                    float weightW = input.W - (int)input.W;

                    //calc biLinear weighted point
                    biP1P3 = origImg.at<Vec3b>(input.H, input.W)[k] * (1 - weightH) +
                        origImg.at<Vec3b>(input.H + 1, input.W)[k] * weightH;
                    biP1P2 = origImg.at<Vec3b>(input.H, input.W)[k] * (1 - weightW) +
                        origImg.at<Vec3b>(input.H, input.W + 1)[k] * weightW;
                    biP3P4 = origImg.at<Vec3b>(input.H + 1, input.W)[k] * (1 - weightW) +
                        origImg.at<Vec3b>(input.H + 1, input.W + 1)[k] * weightW;
                    biP2P4 = origImg.at<Vec3b>(input.H, input.W + 1)[k] * (1 - weightH) +
                        origImg.at<Vec3b>(input.H + 1, input.W + 1)[k] * weightH;

                    //save weighted point at outputImg
                    outputImg->at<Vec3b>(i, j)[k] = (biP1P3 * (1 - weightW) + biP2P4 * weightW +
                        biP1P2 * (1 - weightH) + biP3P4 * weightH) / 2;
                }
                else {
                    //fill blank by edge pixel
                    /*
                    * outputImg->at<Vec3b>(i, j)[k] = origImg.at<Vec3b>((input.H >= origImg.rows) ? (origImg.rows - 1) : (input.H > 0 ? input.H : 0),
                    *    (input.W >= origImg.cols) ? (origImg.cols - 1) : (input.W > 0 ? input.W : 0))[k];
                    */
                }

            }
        }
    }
}
