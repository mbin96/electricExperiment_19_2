#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include <random>
#include <ctime>
#define KMEANS 16

struct pixel {
    uchar R;
    uchar G;
    uchar B;
    float x = -1;
    float y = -1;
    int label;
}typedef pixel;

pixel * centerCreate(int * count, int k ) {

    pixel* center = (pixel*)malloc(sizeof(pixel) * k * k * k);
    (*count) = KMEANS;
    srand((unsigned int)time(0));

    
    for (int r = 0; r < KMEANS; r++) {
        center[r].R =
            rand() % 256;
        center[r].G =
            rand() % 256;
        center[r].B =
            rand() % 256;
        center[r].label = *count;
            
    }
    return center;
}

pixel * mapImg(cv::Mat img) {

    pixel* imgpixel = (pixel*)calloc(img.cols * img.rows, sizeof(pixel));
    for (int h = 0; h < img.cols; h++) {
        for (int w = 0; w < img.rows; w++) {

            imgpixel[h * img.rows + w].x = w;
            imgpixel[h * img.rows + w].y = h;
            imgpixel[h * img.rows + w].R = img.at<cv::Vec3b>(h, w)[2];
            imgpixel[h * img.rows + w].G = img.at<cv::Vec3b>(h, w)[1];
            imgpixel[h * img.rows + w].B = img.at<cv::Vec3b>(h, w)[0];
        }
    }
    return imgpixel;
}


pixel* kmeancalc(cv::Mat img, pixel * center, pixel * imgpixel,int kCount) {
    float* dest = (float*)calloc(kCount, sizeof(int));
    float min = INT_MAX, minCenter;
    
    for (int h = 0; h < img.cols; h++) {
        for (int w = 0; w < img.rows; w++) {
            min = INT_MAX;
            for (int c = 0; c < kCount; c++) {
                dest[c] = sqrt(pow(center[c].B - imgpixel[h * img.rows + w].B, 2) +
                    pow(center[c].G - imgpixel[h * img.rows + w].G, 2) +
                    pow(center[c].R - imgpixel[h * img.rows + w].R, 2));
                if (min > dest[c]) {
                    min = dest[c];
                    minCenter = c;
                }


            }

            imgpixel[h * img.rows + w].label = minCenter;
        }
    }

    for (int c = 0; c < kCount; c++) {

        float sumR = 0, sumG = 0, sumB = 0;
        int count = 0;

        for (int h = 0; h < img.cols; h++) {
            for (int w = 0; w < img.rows; w++) {
                if ((imgpixel[h * img.rows + w].label) == c) {
                    sumR += imgpixel[h * img.rows + w].R;

                    sumG += imgpixel[h * img.rows + w].G;

                    sumB += imgpixel[h * img.rows + w].B;
                    count++;
                }
            }
        }

        center[c].R = sumR / count;
        center[c].G = sumG / count;
        center[c].B = sumB / count;

    }

    return center;
}

int main() {
    int kCount;
    pixel * center = centerCreate(&kCount, KMEANS);
    cv::Mat img = cv::imread("rena.bmp", cv::IMREAD_COLOR);
    float * dest = (float *)calloc(kCount,sizeof(int));
    float min = INT_MAX, minCenter;
    
    pixel * imgpixel = mapImg(img);
    for (int h = 0; h < img.cols; h++) {
        for (int w = 0; w < img.rows; w++) {
            min = INT_MAX;
            for (int c = 0; c < kCount; c++) {
                dest[c] = sqrt(pow(center[c].B - imgpixel[h * img.rows + w].B, 2) +
                    pow(center[c].G - imgpixel[h * img.rows + w].G, 2) +
                    pow(center[c].R - imgpixel[h * img.rows + w].R, 2));
                if (min > dest[c]) {
                    min = dest[c];
                    minCenter = c;
                }
                

            }

            imgpixel[h * img.rows + w].label = minCenter;
        }
    }
    
    for (int c = 0; c < kCount; c++) {

        float sumR = 0, sumG = 0, sumB = 0;
        int count=0;

        for (int h = 0; h < img.cols; h++) {
            for (int w = 0; w < img.rows; w++) {
                if ((imgpixel[h * img.rows + w].label) == c) {
                    sumR += imgpixel[h * img.rows + w].R;

                    sumG += imgpixel[h * img.rows + w].G;

                    sumB += imgpixel[h * img.rows + w].B;
                    count++;
                }
            }
        }

        center[c].R = sumR / count;
        center[c].G = sumG / count;
        center[c].B = sumB / count;

    }

    for (int i = 0; i < 20; i++) {

        pixel* centertmp = kmeancalc(img, center, imgpixel, kCount);
        center = kmeancalc(img, centertmp, imgpixel, kCount);
    }



    cv::Mat imgout = cv::Mat::zeros(img.cols, img.rows, CV_8UC3);
    for (int h = 0; h < img.cols; h++) {
        for (int w = 0; w < img.rows; w++) {
            imgout.at<cv::Vec3b>(h, w)[2] = center[imgpixel[h * img.rows + w].label].R;
            imgout.at<cv::Vec3b>(h, w)[1] = center[imgpixel[h * img.rows + w].label].G;
            imgout.at<cv::Vec3b>(h, w)[0] = center[imgpixel[h * img.rows + w].label].B;
        }
    }
    imshow("result", imgout);
    cv::waitKey(5000);
}