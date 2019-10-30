#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#define USEINTMAP


//header
#include "visionFunctions.hpp"
#define HOG_BLK_SIZE 18

void hog_point_1(pixel* output, int y, int x, int tilesize, int height, int width, float * hogWeight) {
	int tileW, tileH, blkI, blkJ, sumBlk = 0;
	tileW = tileH = tilesize;
	int i, j, h, w;
	int startX = x - (int)(tileW / 2), startY = y - (int)(tileH / 2);
	float quantPhase = 0, phaseWeight = 0;
	for (h = 0; h < tileH; h++) {
		for (w = 0; w < tileW; w++) {
			//countinue when outbound
			if ((h + startY < 0) || (h + startY >= height)) continue;
			if ((w + startX < 0) || (w + startX >= width)) continue;
			//calc hog weight
			quantPhase = output[(h + startY) * width + (w + startX)].phase / 20;
			phaseWeight = quantPhase - (int)quantPhase;
			hogWeight[(int)quantPhase] +=
				(1 - phaseWeight) * output[(h + startY) * width + (w + startX)].magnitude;
			hogWeight[(int)(quantPhase + 1 >= 9 ? 0 : quantPhase + 1)] +=
				(phaseWeight)* output[(h + startY) * width + (w + startX)].magnitude;
		}
	}

	//sum of all block pixel's magnitude
	for (int k = 0; k < 9; k++) {
		sumBlk += pow(hogWeight[k], 2);
	}

	//L-2 normalize weight
	for (int k = 0; k < 9; k++) {
		hogWeight[k] = hogWeight[k] / sqrt(sumBlk) + 0.000001;
	}
}



float ** hogSumImg(pixel * weight,int height,int width) {
	float ** sum = (float **) calloc(height * width, sizeof(float *));
	int h, w, hh, ww;
	float quantPhase, phaseWeight;
	//calc hog
	for (h = 0; h < height; h++) {
		for (w = 0; w < width; w++) {
			sum[h * width + w] = (float*)calloc(9, sizeof(float));
			//calc hog weight
			quantPhase = weight[h * width + w].phase / 20;
			phaseWeight = quantPhase - (int)quantPhase;
			sum[h * width + w][(int)quantPhase] +=
				(1 - phaseWeight) * weight[h * width + w].magnitude;
			sum[h * width + w][(int)(quantPhase + 1 >= 9 ? 0 : quantPhase + 1)] +=
				(phaseWeight)* weight[h * width + w].magnitude;
			if (h == 0 && w == 0) {

			}
			else if (h == 0) {
				for (int k = 0; k < 9; k++) {
					sum[h * width + w][k] = sum[h * width + w][k] + sum[h * width + w - 1][k];
				}
			}
			else if (w == 0) {
				for (int k = 0; k < 9; k++) {
					sum[h * width + w][k] = sum[h * width + w][k] + sum[(h - 1) * width + w][k];
				}
			}
			else {
				for (int k = 0; k < 9; k++) {
					sum[h * width + w][k] = sum[h * width + w][k] + sum[(h - 1) * width + w][k] + sum[h * width + w - 1][k] - sum[(h - 1) * width + w - 1][k];
				}
			}
		}
	}
	return sum;
}

//get hog weight from intmap
//x,y to x+size,y+size
float * getHogFromMap(float ** intmap, int size, int x, int y, int width, int height) {
	float * hogWeight = (float *)calloc(9,sizeof(float));
	float sumBlk=0;
	for (int k = 0; k < 9; k++) {
		hogWeight[k] = intmap[expendEdge(y,height) * width + expendEdge(x,width)][k] - intmap[expendEdge(y, height) * width + expendEdge(x + size, width)][k] - intmap[expendEdge(y + size, height) * width + expendEdge(x,width)][k] + intmap[expendEdge(y + size,height) * width + expendEdge(x + size,width)][k];
	}

	for (int k = 0; k < 9; k++) {
		sumBlk += pow(hogWeight[k], 2);
	}

	//L-2 normalize weight
	for (int k = 0; k < 9; k++) {
		hogWeight[k] = hogWeight[k] / (sqrt(sumBlk) + 0.000001);
	}
	return hogWeight;
}

int main() {
	
	Mat sampleFace = imread("ref.bmp", cv::IMREAD_COLOR);
	Mat targetFace = imread("tar.bmp", cv::IMREAD_COLOR);
	pixel* weightSampleFace = setVision(sampleFace, GAU_DISABLE, HOG_DISABLE);
	pixel* weightTargetFace = setVision(targetFace, GAU_DISABLE, HOG_DISABLE);
	FILE * fp = fopen("hell.txt","w");

	int height = sampleFace.rows;
	int width = sampleFace.cols;
	int i, j, k;
	

#ifdef USEINTMAP

	

	float* hogWeightRef[9], * hogWeightTar[9];//81차원
	for (k = 0; k < 9; k++) {
		hogWeightRef[k] = (float*)calloc(9, sizeof(float));

	}
	k = 0;
	for (i = 0; (i + 1) * (HOG_BLK_SIZE / 2) < height; i++) {
		for (j = 0; (j + 1) * (HOG_BLK_SIZE / 2) < width; j++) {

			hog_point_1(weightSampleFace, (i + 1) * (HOG_BLK_SIZE / 2), (j + 1) * (HOG_BLK_SIZE / 2), HOG_BLK_SIZE, height, width, hogWeightRef[k++]);

		}
	}
#ifdef DEBUG
	std::ofstream out("ee.txt");
	for (i = 0; i < 9; i++) {
		for (j = 0; j < 9; j++) {
			out << std::to_string(hogWeightRef[i][j]) << std::endl;
		}
	}
#endif DEBUG

	height = targetFace.rows;
	width = targetFace.cols;
	//calculation intmap
	float ** intmap = hogSumImg(weightTargetFace, height, width);

	float max = INT_MIN, min = INT_MAX;
	int ii, jj, l;
	for (ii = 0; ii < height; ii++) {
		for (jj = 0; jj < width; jj++) {
			
			//demension
			for (i = 0; i < 3; i++) {
				for (j = 0; j < 3; j++) {
					hogWeightTar[i*3+j] = (float *)getHogFromMap(intmap, HOG_BLK_SIZE, jj - HOG_BLK_SIZE + j * (HOG_BLK_SIZE / 2), ii - HOG_BLK_SIZE + i * (HOG_BLK_SIZE / 2), width, height);
					for (l = 0; l < 9; l++) {
						weightTargetFace[ii * width + jj].face += fabs(hogWeightRef[i * 3 + j][l] - hogWeightTar[i * 3 + j][l]);
					}
					free(hogWeightTar[i * 3 + j]);
				}
			}

			if (max < weightTargetFace[ii * width + jj].face)
				max = weightTargetFace[ii * width + jj].face;
			if (min > weightTargetFace[ii * width + jj].face)
				min = weightTargetFace[ii * width + jj].face;
		}
		std::cout << ii << std::endl;
	}
	for (int h = 0; h < height; h++) {
		for (int w = 0; w < width; w++) {
			free(intmap[h * width + w]);
		}
	}
#else
	float* hogWeightRef[9], * hogWeightTar[9];//81차원
	for (k = 0; k < 9; k++) {
		hogWeightRef[k] = (float*)calloc(9, sizeof(float));

	}
	k = 0;
	for (i = 0; (i + 1) * (HOG_BLK_SIZE / 2) < height; i++) {
		for (j = 0; (j + 1) * (HOG_BLK_SIZE / 2) < width; j++) {

			hog_point_1(weightSampleFace, (i + 1) * (HOG_BLK_SIZE / 2), (j + 1) * (HOG_BLK_SIZE / 2), HOG_BLK_SIZE, height, width, hogWeightRef[k++]);

		}
	}
	std::ofstream out("ee.txt");
	for (i = 0; i < 9; i++) {
		for (j = 0; j < 9; j++) {


			out << std::to_string(hogWeightRef[i][j]) << std::endl;

		}
	}

	height = targetFace.rows;
	width = targetFace.cols;
	float max = INT_MIN, min = INT_MAX;
	int ii, jj, l;
	for (ii = 0; ii < height; ii++) {
		for (jj = 0; jj < width; jj++) {
			for (int k = 0; k < 9; k++) {//dimension
				hogWeightTar[k] = (float*)calloc(9, sizeof(float));
			}
			k = 0;
			for (i = 0; i < 3; i++) {
				for (j = 0; j < 3; j++) {
					hog_point_1(weightTargetFace, ii - HOG_BLK_SIZE + (i + 1) * (HOG_BLK_SIZE / 2), jj - HOG_BLK_SIZE + (j + 1) * (HOG_BLK_SIZE / 2), HOG_BLK_SIZE, height, width, hogWeightTar[k]);
					for (l = 0; l < 9; l++)
						weightTargetFace[ii * width + jj].face += fabs(hogWeightRef[k][l] - hogWeightTar[k][l]);
					k++;
				}
			}

			if (max < weightTargetFace[ii * width + jj].face)
				max = weightTargetFace[ii * width + jj].face;
			if (min > weightTargetFace[ii * width + jj].face)
				min = weightTargetFace[ii * width + jj].face;
		}
		std::cout << ii << std::endl;
	}
#endif //useintmap




	Mat output = Mat::zeros(height, width, CV_8UC1);

	
	for ( ii = 0; ii < height; ii++) {
		for ( jj = 0; jj < width; jj++) {
			output.at<uchar>(ii, jj) = (weightTargetFace[ii * width + jj].face-min)*255/(max - min +0.0000001);
			if (1 < output.at<uchar>(ii, jj) && 35 > output.at<uchar>(ii, jj)) {
				Point lb(jj + 25, ii + 25);
				Point tr(jj-25, ii-25);
				rectangle(targetFace, lb, tr, Scalar(0, 255, 0), 3, 8, 0);
			}
		}
	}
	

	imwrite("reasult.bmp", targetFace);
	imshow("reasult", targetFace);
	imwrite("result.bmp", output);
	imshow("result", output);
	waitKey(5000);
	
	

	


	/*
	//calc vision weight of ref Img.
	Mat refImg = imread("ref.bmp", cv::IMREAD_COLOR);
	height = refImg.rows;
	width = refImg.cols;
	pixel* visionWeightRef = setVision(refImg);

	//calc vision weight of compare Img.
	Mat compImg = imread("tarShift_1.bmp", cv::IMREAD_COLOR);
	height = compImg.rows;
	width = compImg.cols;
	pixel* visionWeightComp = setVision(compImg);

	findSameEdge(refImg, compImg, visionWeightRef, visionWeightComp);
	waitKey(5000);
	*/

}

