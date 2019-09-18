#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#pragma warning(disable : 4996)

using namespace cv;

void main() {
	Mat img = imread("photo.jpg", cv::IMREAD_COLOR);

	//input position
	float scale;
	for (;;) {
		printf("input scale > ");
		scanf("%f", &scale);
		if (scale>0) {
			break;
		}
	}

	//calc output size and scale
	int outputH = img.rows * scale;
	int outputW = img.cols * scale;
	float scaleInv = 1 / scale;

	//dical. scaled Img
	Mat imgScaled = Mat::zeros(outputH, outputW, CV_8UC3);

	/* pixel tile
	*  p1-------p2
	*	|		|
	*	|		|
	*	|		|
	*  p3-------p4
	*/

	float biP1P3, biP1P2, biP3P4, biP2P4;
	float inputH, inputW;
	//Y
	for (int i = 0; i < outputH; i++) {
		inputH = i * scaleInv;
		//X
		for (int j = 0; j < outputW; j++) {
			inputW = j * scaleInv;
			for (int k = 0; k < 3; k++) {
				if (inputH + 1 < img.rows && inputW + 1 < img.cols) {
					//calc horize and portrait weight
					biP1P3 = img.at<Vec3b>(inputH, inputW)[k] * (1 - (inputH - (int)inputH)) +
						img.at<Vec3b>(inputH + 1, inputW)[k] * (inputH - (int)inputH);
					biP1P2 = img.at<Vec3b>(inputH, inputW)[k] * (1 - (inputW - (int)inputW)) +
						img.at<Vec3b>(inputH, inputW + 1)[k] * (inputW - (int)inputW);
					biP3P4 = img.at<Vec3b>(inputH + 1, inputW)[k] * (1 - (inputW - (int)inputW)) +
						img.at<Vec3b>(inputH + 1, inputW + 1)[k] * (inputW - (int)inputW);
					biP2P4 = img.at<Vec3b>(inputH, inputW + 1)[k] * (1 - (inputH - (int)inputH)) +
						img.at<Vec3b>(inputH + 1, inputW + 1)[k] * (inputH - (int)inputH);

					//save
					imgScaled.at<Vec3b>(i, j)[k] = (biP1P3 * (1 - (inputW - (int)inputW)) + biP2P4 * (inputW - (int)inputW) +
						biP1P2 * (1 - (inputH - (int)inputH)) + biP3P4 * (inputH - (int)inputH))/2;
				} else {

				}
			}
		}
	}

	//show result
	imshow("result", imgScaled);
	imshow("orig", img);
	waitKey(5000);
	imwrite("output.bmp", imgScaled);
}

