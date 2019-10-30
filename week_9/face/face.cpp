#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <math.h>
#include <iostream>
#include <stdlib.h>


using namespace cv;
using namespace cv::ml;
using namespace std;

#define  LBP_INPUT_SIZE 128
//header
//#include "visionFunctions.hpp"

Mat lbpImg(Mat origImg) {
	int i, j, h, w,ii,jj;
    float max = INT_MIN;

	Mat origImgGray = Mat::zeros(LBP_INPUT_SIZE, LBP_INPUT_SIZE, CV_8UC1);
	for (i = 0; i < LBP_INPUT_SIZE; i++) {
		//X
		for (j = 0; j < LBP_INPUT_SIZE; j++) {
			origImgGray.at<uchar>(i, j) = (origImg.at<Vec3b>(i, j)[0] + origImg.at<Vec3b>(i, j)[1] + origImg.at<Vec3b>(i, j)[2]) / 3;
		}
	}
	int tileSize = LBP_INPUT_SIZE / 4;
	Mat outImg = Mat::zeros(LBP_INPUT_SIZE, LBP_INPUT_SIZE, CV_8UC1);
	unsigned char tmp = 0;
	for (h = 1; h < LBP_INPUT_SIZE - 1; h++) {
		//X
		for (w = 1; w < LBP_INPUT_SIZE - 1; w++) {
			tmp = 0;

			if( (origImgGray.at<uchar>(h, w) < origImgGray.at<uchar>(h - 1, w + 0))) tmp +=1;
			if( (origImgGray.at<uchar>(h, w) < origImgGray.at<uchar>(h - 1, w + 1))) tmp +=2;
			if( (origImgGray.at<uchar>(h, w) < origImgGray.at<uchar>(h - 0, w + 1))) tmp +=4;
			if( (origImgGray.at<uchar>(h, w) < origImgGray.at<uchar>(h + 1, w + 1))) tmp +=8;
			if( (origImgGray.at<uchar>(h, w) < origImgGray.at<uchar>(h + 1, w + 0))) tmp +=16;
			if( (origImgGray.at<uchar>(h, w) < origImgGray.at<uchar>(h + 1, w - 1))) tmp +=32;
			if( (origImgGray.at<uchar>(h, w) < origImgGray.at<uchar>(h + 0, w - 1))) tmp +=64;
			if( (origImgGray.at<uchar>(h, w) < origImgGray.at<uchar>(h - 1, w - 1))) tmp +=128;

			outImg.at<uchar>(h, w) = tmp;
		}
	}
	
	return outImg;//false face
}

int lbpComp(Mat ref, Mat tar) {
	int h, w, i,j;
	int tileH = LBP_INPUT_SIZE / 4, tileW = LBP_INPUT_SIZE / 4;
	float similality = 0;
	float * lbpDi;
	float * lbpDiTar;

	for (j = 0; j < LBP_INPUT_SIZE - 16; j += 16) {
		for (i = 0; i < LBP_INPUT_SIZE - 16; i += 16) {
			lbpDi = (float *)calloc(256, sizeof(float));
			lbpDiTar = (float *)calloc(256, sizeof(float));
			
			for (h = 0; h < tileH; h++) {
				for (w = 0; w < tileW; w++) {
					lbpDi[ref.at<uchar>(i + h, j + w)] = 1;
					lbpDiTar[tar.at<uchar>(i + h, j + w)] = 1;
				}
			}
			for (h = 0; h < 256; h++) {
				similality += fabs(lbpDi[h] - lbpDiTar[h]) / 256;
			}
			free(lbpDi);
			free(lbpDiTar);
		}
	}

	if (similality < 10) {
		return 1;//same face
	}
	else {
		return 0;//dif face
	}
}

int main() {

	CascadeClassifier cascadeFace;
	cascadeFace.load("C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml");

	VideoCapture capture(0);

	if (!capture.isOpened()) {
		printf("WTF");
		return 0;
	}

	Mat frameRef;

	capture >> frameRef;

	//Mat faceOrig = imread("faceTar.bmp", 1);
	vector<Rect> faces;
	Mat lbpRef = Mat::zeros(LBP_INPUT_SIZE, LBP_INPUT_SIZE, CV_8UC1);
	Mat lbpTar = Mat::zeros(LBP_INPUT_SIZE, LBP_INPUT_SIZE, CV_8UC1);
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
					faceDetected.at<Vec3b>(h-faces[y].y,w- faces[y].x)[0] = (frameRef.at<Vec3b>(h,w)[0]);
					faceDetected.at<Vec3b>(h-faces[y].y,w- faces[y].x)[1] = (frameRef.at<Vec3b>(h,w)[1]);
					faceDetected.at<Vec3b>(h-faces[y].y,w- faces[y].x)[2] = (frameRef.at<Vec3b>(h,w)[2]);
				}
			}

			resize(faceDetected, faceDetected, Size(LBP_INPUT_SIZE, LBP_INPUT_SIZE));
			lbpTar=lbpImg(faceDetected);
			//imshow("img", lbpTar);
			if (lbpComp(lbpRef, lbpTar)) {
				rectangle(frameRef, lb, tr, Scalar(0, 255, 0), 3, 8, 0);
			}
			else {
				rectangle(frameRef, lb, tr, Scalar(0, 0, 255), 3, 8, 0);
			}
		}

		imshow("img", frameRef);
		
		if (waitKey(30) >= 0) {
			lbpRef = lbpTar;

		}
	}

}

