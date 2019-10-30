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


//header
//#include "visionFunctions.hpp"

int main() {

	CascadeClassifier cascadeFace, cascadeSmail;
	cascadeFace.load("C:/opencv/sources/data/lbpcascades/lbpcascade_frontalface_improved.xml");

	VideoCapture capture(0);

	if (!capture.isOpened()) {
		printf("ee");
		return 0;
	}

	Mat frameRef;


	capture >> frameRef;

	//Mat faceOrig = imread("faceTar.bmp", 1);
	vector<Rect> faces;

	while (true) {
		capture >> frameRef;
		cascadeFace.detectMultiScale(frameRef, faces, 1.1, 4, 0 | 2, Size(10, 10));

		for (int y = 0; y < faces.size(); y++) {
			Point lb(faces[y].x + faces[y].width, faces[y].y + faces[y].height);
			Point tr(faces[y].x, faces[y].y);
			rectangle(frameRef, lb, tr, Scalar(0, 255, 0), 3, 8, 0);
		}
		imshow("Face", frameRef);
		waitKey(30);
	}


	
	/*
	//open img
	int height, width;
	pixel* frameWeightRef, * frameWeightTar;
	VideoCapture capture(0);
	

	if (!capture.isOpened()) {
		printf("�ƾ�");
		return 0;
	}

	Mat frameRef;

	height = frameRef.rows;
	width = frameRef.cols;
	Mat frame;
	capture >> frameRef;
	frameWeightRef = setVision(frameRef, GAU_DISABLE, HOG_ENABLE);

	while (true) {

		capture >> frame;
		frameWeightTar = setVision(frame, GAU_DISABLE, HOG_ENABLE);
		imshow("video", findSameEdge(frameRef, frame, frameWeightRef, frameWeightTar));

		if (waitKey(1) >= 0) {
			capture >> frameRef;
			free(frameWeightRef);
			frameWeightRef = setVision(frameRef, GAU_DISABLE, HOG_ENABLE);
		}
		free(frameWeightTar);
	}
	*/
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

