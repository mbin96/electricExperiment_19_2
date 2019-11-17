


#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "ldmarkmodel.h"

using namespace std;
using namespace cv;

#define  LBP_INPUT_SIZE 16
#define LBP_DIMENSION 30

Mat lbpCut(Mat origImg, int x, int y) {

	//들어온 좌표를 가운데로 하기위해 빼주기
	int startX = x - LBP_INPUT_SIZE / 2;
	int startY = y - LBP_INPUT_SIZE / 2;

	int h, w;

	//outputimg
	Mat outImg = Mat::zeros(LBP_INPUT_SIZE, LBP_INPUT_SIZE, CV_8UC3);

	for (h = 0; h < LBP_INPUT_SIZE; h++) {
		for (w = 0; w < LBP_INPUT_SIZE; w++) {
			outImg.at<Vec3b>(h, w) = origImg.at<Vec3b>(startY + h, startX + w);
		}
	}
	return outImg;
}


Mat lbpImg(Mat origImg) {
	int i, j, h, w, ii, jj;
	float max = INT_MIN;

	Mat origImgGray = Mat::zeros(LBP_INPUT_SIZE, LBP_INPUT_SIZE, CV_8UC1);
	for (i = 0; i < LBP_INPUT_SIZE; i++) {
		//X
		for (j = 0; j < LBP_INPUT_SIZE; j++) {
			origImgGray.at<uchar>(i, j) = (origImg.at<Vec3b>(i, j)[0] + origImg.at<Vec3b>(i, j)[1] + origImg.at<Vec3b>(i, j)[2]) / 3;
		}
	}

	Mat outImg = Mat::zeros(LBP_INPUT_SIZE, LBP_INPUT_SIZE, CV_8UC1);
	unsigned char tmp = 0;
	for (h = 1; h < LBP_INPUT_SIZE - 1; h++) {
		//X
		for (w = 1; w < LBP_INPUT_SIZE - 1; w++) {
			tmp = 0;

			if ((origImgGray.at<uchar>(h, w) < origImgGray.at<uchar>(h - 1, w + 0))) tmp += 1;
			if ((origImgGray.at<uchar>(h, w) < origImgGray.at<uchar>(h - 1, w + 1))) tmp += 2;
			if ((origImgGray.at<uchar>(h, w) < origImgGray.at<uchar>(h - 0, w + 1))) tmp += 4;
			if ((origImgGray.at<uchar>(h, w) < origImgGray.at<uchar>(h + 1, w + 1))) tmp += 8;
			if ((origImgGray.at<uchar>(h, w) < origImgGray.at<uchar>(h + 1, w + 0))) tmp += 16;
			if ((origImgGray.at<uchar>(h, w) < origImgGray.at<uchar>(h + 1, w - 1))) tmp += 32;
			if ((origImgGray.at<uchar>(h, w) < origImgGray.at<uchar>(h + 0, w - 1))) tmp += 64;
			if ((origImgGray.at<uchar>(h, w) < origImgGray.at<uchar>(h - 1, w - 1))) tmp += 128;

			outImg.at<uchar>(h, w) = tmp;
		}
	}

	return outImg;//false face
}

int lbpComp(Mat ref, Mat tar, int dimension) {
	int h, w, i, j;
	int tileH = LBP_INPUT_SIZE, tileW = LBP_INPUT_SIZE;

	float similality = 0;

	float* lbpDi;
	float* lbpDiTar;

	for (int di = 0; di < dimension; di++) {
		lbpDi = (float*)calloc(256, sizeof(float));
		lbpDiTar = (float*)calloc(256, sizeof(float));

		for (h = 0; h < tileH; h++) {
			for (w = 0; w < tileW; w++) {
				lbpDi[ref.at<uchar>(di * tileH + h, w)] += 1;
				lbpDiTar[tar.at<uchar>(di * tileH + h, w)] += 1;
			}
		}

		for (h = 0; h < 256; h++) {
			similality += fabs(lbpDi[h] - lbpDiTar[h]) / 256;
		}
		free(lbpDi);
		free(lbpDiTar);

	}
	similality /= dimension;

	if (similality < 0.1) {
		return 1;//same face
	}
	else {
		return 0;//dif face
	}
}

int main()
{
	/*********************
	std::vector<ImageLabel> mImageLabels;
	if(!load_ImageLabels("mImageLabels-test.bin", mImageLabels)){
		mImageLabels.clear();
		ReadLabelsFromFile(mImageLabels, "labels_ibug_300W_test.xml");
		save_ImageLabels(mImageLabels, "mImageLabels-test.bin");
	}
	std::cout << "what: " <<  mImageLabels.size() << std::endl;
	*******************/

	ldmarkmodel modelt;
	std::string modelFilePath = "roboman-landmark-model.bin";
	while (!load_ldmarkmodel(modelFilePath, modelt)) {
		std::cout << "Load error." << std::endl;
		std::cin >> modelFilePath;
	}

	//cv::VideoCapture mCamera(0);
	cv::VideoCapture mCamera("head-pose-face-detection-male.mp4");
	if (!mCamera.isOpened()) {
		std::cout << "Camera opening failed..." << std::endl;
		system("pause");
		return 0;
	}

	cv::Mat Image;
	cv::Mat current_shape;

	cv::Mat refImg;
	//ref
	mCamera >> refImg;

	//init
	modelt.track(refImg, current_shape);
	cv::Vec3d eav;
	modelt.EstimateHeadPose(current_shape, eav);
	modelt.drawPose(refImg, current_shape, 50);
	int numLandmarks = current_shape.cols / 2;

	Mat reflbp = Mat::zeros(LBP_INPUT_SIZE * LBP_DIMENSION, LBP_INPUT_SIZE, CV_8UC3);
	Mat tarlbp = Mat::zeros(LBP_INPUT_SIZE * LBP_DIMENSION, LBP_INPUT_SIZE, CV_8UC3);
	Mat tile;
	for (int j = 16; j < numLandmarks; j++) {
		int x = current_shape.at<float>(j);
		int y = current_shape.at<float>(j + numLandmarks);
		std::stringstream ss;
		ss << j;

		tile = lbpImg(lbpCut(refImg, x, y));

		for (int h = 0; h < LBP_INPUT_SIZE; h++) {
			for (int w = 0; w < LBP_INPUT_SIZE; w++) {
				reflbp.at<Vec3b>((j - 16) * LBP_INPUT_SIZE + h, w) = tile.at<Vec3b>(h, w);
			}
		}
		//            cv::putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 0, 255));
		cv::circle(Image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
	}

	for (;;) {
		mCamera >> Image;
		modelt.track(Image, current_shape);
		cv::Vec3d eav;
		modelt.EstimateHeadPose(current_shape, eav);
		modelt.drawPose(Image, current_shape, 50);

		int numLandmarks = current_shape.cols / 2;
		for (int j = 0; j < 16; j++) {
			int x = current_shape.at<float>(j);
			int y = current_shape.at<float>(j + numLandmarks);
			std::stringstream ss;
			ss << j;
			//            cv::putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 0, 255));
			cv::circle(Image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
		}

		for (int j = 16; j < numLandmarks; j++) {
			int x = current_shape.at<float>(j);
			int y = current_shape.at<float>(j + numLandmarks);
			std::stringstream ss;
			ss << j;

			Mat tile = lbpImg(lbpCut(refImg, x, y));

			for (int h = 0; h < LBP_INPUT_SIZE; h++) {
				for (int w = 0; w < LBP_INPUT_SIZE; w++) {
					tarlbp.at<Vec3b>((j - 16) * LBP_INPUT_SIZE + h, w) = tile.at<Vec3b>(h, w);
				}
			}

			//            cv::putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 0, 255));
			cv::circle(Image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
		}

		if (lbpComp(reflbp, tarlbp, numLandmarks - 16) == 1) {
			cv::putText(Image, "너다", cv::Point(10, 10), 1, 2, cv::Scalar(0, 0, 255));
		}

		cv::imshow("Camera", Image);
		int key = waitKey(5);
		if (27 == key) {
			mCamera.release();
			cv::destroyAllWindows();
			break;
		}
		else if (key >= 0) {
			reflbp = tarlbp;
		}


	}

	system("pause");
	return 0;
}














































