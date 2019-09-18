#include <opencv2/imgproc.hpp>
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

#define DEBUG_

struct pixel {
	float H;
	float W;
	float phase;
	float magnitude;
}typedef pixel;

using namespace cv;

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

//void convolution(Mat* outputImg, Mat origImg);

int main() {
	//open img
	Mat origImg = imread("ref6.jpg", cv::IMREAD_COLOR);
	int height = origImg.rows;
	int width = origImg.cols;
	int i, j,h,w;
	//Mat origImgGray(height, width, CV_8UC1);
	Mat origImgGray = Mat::zeros(height, width, CV_8UC1);
	for (i = 0; i < height; i++) {
		//X
		for (j = 0; j < width; j++) {
			origImgGray.at<uchar>(i, j) = (origImg.at<Vec3b>(i, j)[0] + origImg.at<Vec3b>(i, j)[1] + origImg.at<Vec3b>(i, j)[2]) / 3;
		}
	}
	Mat outputImg(height, width, CV_8UC1);

	pixel* output = (pixel*)calloc(height * width, sizeof(pixel));
	
	int max = INT_MIN, min = INT_MAX ;
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
					//calc MAC gradiant filter
					output[i * width + j].W += (origImgGray.at<uchar>(i - 1 + h, j - 1 + w)) * (filterX[h * FILTER_W + w]);
					output[i * width + j].H += (origImgGray.at<uchar>(i - 1 + h, j - 1 + w)) * (filterY[h * FILTER_W + w]);
				}
			}

			//calc phase
			output[i * width + j].phase = 180 * (atan2(output[i * width + j].H, output[i * width + j].W) / PI);
			

			if (output[i * width + j].phase < 0) {
				output[i * width + j].phase = output[i * width + j].phase + 180;
			}
			if (output[i * width + j].phase >= 180) {
				output[i * width + j].phase = 0;
			}
			//Phase quantization by 20 degree
			output[i * width + j].phase = (int)(output[i * width + j].phase / 20);

			//calc magnitude
			output[i * width + j].magnitude = sqrt(pow(output[i * width + j].W,2) + pow(output[i * width + j].H,2));
			
			//set max, min for normalize
			max = __max(output[i * width + j].magnitude, max);
			min = __min(output[i * width + j].magnitude, min);
		}
	}

	//magnitude normalize
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			output[i * width + j].magnitude = 255 * (output[i * width + j].magnitude - min) / (max - min);
			outputImg.at<uchar>(i, j) = output[i * width + j].magnitude;
		}
	}

	int tileW = 16, tileH = 16, blkI, blkJ, sumBlk = 0;
	float block[15][7][9] = { 0, };

	for (i = 0; i <= height - tileH; i += 8) {
		for (j = 0; j <= width - tileW; j += 8) {
			blkI = i / 8;
			blkJ = j / 8;
#ifdef DEBUG
			std::cout << i << "," << j << std::endl;
#endif // DEBUG

			//calculation sum of demention's magnitude
			for (h = 0; h < tileH; h++){
				for (w = 0; w < tileW; w++) {
					block[blkI][blkJ][((int)output[(h + i) * width + (w + j)].phase)] += 
						output[(h+i)* width + (w+j)].magnitude;
				}
			}

			for (int k = 0; k<9; k++) {
				sumBlk += pow(block[blkI][blkJ][k],2) + 0.000001;
			}

			//L-2 norm.
			for (int k = 0; k < 9; k++) {
				block[blkI][blkJ][k] = block[blkI][blkJ][k] / sqrt(sumBlk);
			}
			sumBlk = 0;
		}
	}
	FILE * file = fopen("output.csv", "w");
	fprintf(file, ",\n");
	for (i = 0; i <= height - tileH; i += 8) {
		for (j = 0; j <= width - tileW; j += 8) {
			blkI = i / 8;
			blkJ = j / 8;
			for (int k = 0; k < 9; k++) {

				fprintf(file, "%f,\n", block[blkI][blkJ][k]);
			}
		}
	}
#ifdef DEBUG
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			std::cout << output[i * width + j].phase;
		}
		std::cout << std::endl;
	}
#endif // DEBUG

	//show result and save to bmp
	imwrite("output.bmp", outputImg);
	imshow("orig", origImgGray);
	imshow("result", outputImg);
	waitKey(5000);
	//waitKey(5000);


	
	
}
