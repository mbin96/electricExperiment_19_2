#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
#define BLK 50
#pragma warning(disable : 4996)

void main() {
	Mat img = imread("photo.jpg", cv::IMREAD_COLOR);


	//input position
	int posX, posY;
	for (;;) {
		printf("input center pixel > ");
		scanf("%d %d", &posX, &posY);
		//posY = posX = 300;
		//if position is correct -> break
		if (posY < img.rows && posX < img.cols && posX >= 0 && posY >= 0) {
			break;
		}
	}

	//recog. pixel overflow
	int pixelStartX, pixelStartY, pixelEndX, pixelEndY;
	pixelStartX = (posX - BLK >= 0) ? posX - BLK : 0;
	pixelStartY = (posY - BLK >= 0) ? posY - BLK : 0;
	pixelEndX = (posX + BLK < img.cols) ? posX + BLK : img.cols-1;
	pixelEndY = (posX + BLK < img.rows) ? posX + BLK : img.rows-1;

	for (int i = pixelStartY; i< pixelEndY; i++) {
		for (int j = pixelStartX; j < pixelEndX; j++) {
			//calc grayscale data
			uchar convGray = (img.at<Vec3b>(i, j)[2] +
				img.at<Vec3b>(i, j)[1] + img.at<Vec3b>(i, j)[0]) / 3;
			//R
			img.at<Vec3b>(i, j)[2] = convGray;
			//G
			img.at<Vec3b>(i, j)[1] = convGray;
			//B
			img.at<Vec3b>(i, j)[0] = convGray;
		}
	}

	//show result
	imshow("result", img);
	waitKey(5000);
}