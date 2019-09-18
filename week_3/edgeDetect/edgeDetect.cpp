#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <iostream>

#pragma warning(disable : 4996)

//define constant
#define PI 3.1416
#define CHANNEL 3
#define FILTER_H 3
#define FILTER_W 3


struct pixel {
	float H;
	float W;
	float radian;
}typedef pixel;

using namespace cv;

int filterX[9] = {
	-1, -1, -1,
	0,0,0,
	1,1,1
};
int filterY[9] = {
	-1, 0, 1,
	-1,0,1,
	-1,0,1
};

//void convolution(Mat* outputImg, Mat origImg);

int main() {
	//open img
	Mat origImg = imread("bitmap.bmp", cv::IMREAD_COLOR);
	int height = origImg.rows; 
	int width = origImg.cols;
	
	//Mat origImgGray(height, width, CV_8UC1);
	Mat origImgGray = Mat::zeros(height, width, CV_8UC1);
	for (int i = 0; i < height; i++) {
		//X
		for (int j = 0; j < width; j++) {
			origImgGray.at<uchar>(i, j) = (origImg.at<Vec3b>(i, j)[0] + origImg.at<Vec3b>(i, j)[1] + origImg.at<Vec3b>(i, j)[2])/3;
		}
	}
    Mat outputImg(height, width, CV_8UC1);
	

	pixel * output = (pixel *) calloc(height * width, sizeof(pixel));
	//float * outputX;
	//outputX = (float*)calloc(origImgGray.rows * origImgGray.cols, sizeof(float));
	//float * outputY;
	//outputY = (float*)calloc(height * width, sizeof(float));

	//calc like center is 0,0
	//Y
	int max = 0, min = 100000;
	float comp;
	for (int i = 0; i < height; i++) {
		//X
		for (int j = 0; j < width; j++) {
			for (int h = 0; h < FILTER_H; h++) {
				for (int w = 0; w < FILTER_W; w++) {
					if ((i - 1 + h < 0) || (i -1 + h >= height)) continue;
					if ((j - 1 + w < 0) || (j -1 + w >= width)) continue;
					output[i * width + j].W += (origImgGray.at<uchar>(i-1+h, j-1+w)) * (filterX[h * FILTER_W + w]);
					output[i * width + j].H += (origImgGray.at<uchar>(i-1+h, j-1+w)) * (filterY[h * FILTER_W + w]);
				}
			}
			//printf("%d,%d\n", j, i);
			if (output[i * width + j].H == 0) {

			}
			output[i * width + j].radian = atan2(output[i * width + j].H, output[i * width + j].W);
			//printf("%d,%d\n", j, i);
			//outputImg.at<uchar>(i, j) = 255*(((output[i * width + j]).W + (output[i * width + j].H))-min)/(max-min);
			comp = output[i * width + j].W + output[i * width + j].H;
			if (max < comp) {
				max = comp;
			}
			if (min > comp) {
				min = comp;
			}

		}
	}
	for (int i = 0; i < height; i++) {
		//X
		for (int j = 0; j < width; j++) {
			outputImg.at<uchar>(i, j) = 255 * (output[i * width + j].W + output[i * width + j].H - min) / (max - min);
		}
	}


	float tmp = 0;
	imshow("orig", origImgGray);
	imshow("result", outputImg);
	//waitKey(5000);

	for (int i = 0; i < height; i++) {
		//X
		for (int j = 0; j < width; j++) {
			output[i * width + j].radian = 180 * (output[i * width + j].radian / 3.14) ;
			if (output[i * width + j].radian < 0) {
				output[i * width + j].radian = output[i * width + j].radian + 180;
			}
			tmp = output[i * width + j].radian;
			if (tmp < 20) {
				output[i * width + j].radian = 0;
			}else if (tmp < 40) {
				output[i * width + j].radian = 1;
			}else if (tmp < 60) {
				output[i * width + j].radian = 2;
			}else if (tmp < 80) {
				output[i * width + j].radian = 3;
			}else if (tmp < 100) {
				output[i * width + j].radian = 4;
			}else if (tmp < 120) {
				output[i * width + j].radian = 5;
			}else if (tmp < 140) {
				output[i * width + j].radian = 6;
			}else if (tmp < 160) {
				output[i * width + j].radian = 7;
			}else{
				output[i * width + j].radian = 8;
			}
			//if(output[i * width + j].radian != 0)
			std :: cout << output[i * width + j].radian ;
		}
	}
	
	//show result and save to bmp
	
	waitKey(5000);
	imwrite("output.bmp", outputImg);
}

/*
void convolution(Mat* outputImg, Mat origImg) {
	int height = origImg.rows;
	int width = origImg.cols;
	//pixel * output = (pixel *) calloc(height*width, sizeof(pixel));
	float *outputX = (float*)calloc(height * width, sizeof(float));
	float *outputY = (float*)calloc(height * width, sizeof(float));
	//calc like center is 0,0
	//Y
	for (int i = 0; i < height; i++) {
		//X
		for (int j = 0; j < width; j++) {
			for (int h = 0; h < FILTER_H; h++) {
				for (int w = 0; w < FILTER_W; w++) {
					if ((i + h < 0) || (i + h > height + FILTER_H - 2)) continue;
					if ((j + w < 0) || (j + w > width + FILTER_W - 2)) continue;
					outputX[i * height + j] += origImg.at<uchar>(i, j) * filterX[h * FILTER_W + w];
					outputY[i * height + j] += origImg.at<uchar>(i, j) * filterY[h * FILTER_W + w];
				}
			}

			//printf("%d,%d\n", j, i);
			//outputImg->at<uchar>(i, j) = ((output[i * height + j].W) + (output[i * height + j].H))/6;
		}
	}
}
*/
