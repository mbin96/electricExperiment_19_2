#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

using namespace cv;
int main()
{

	cv::VideoCapture cam(0);
	if (!cam.isOpened()) {
		std::cout << "Camera opening failed..." << std::endl;
		system("pause");
		return 0;
	}
	cv::Mat currFrame;
	cam >> currFrame;
	int rectWidth = currFrame.rows * 0.2;
	int preX, preY, currX, currY;
	currY = preY = currFrame.rows / 2 - rectWidth / 2;
	currX = preX = currFrame.cols / 2 - rectWidth / 2;
	Rect rectangle(preX, preY, rectWidth + preX, rectWidth + preY);
	Mat bg, fg;
	cv::Mat1b markers(currFrame.rows, currFrame.cols);
	int veloX = currX - preX;
	int veloY = currY - preY;
	for (;;) {
		cam >> currFrame;
		
		Rect rectangle(currX + veloX, currY + veloY, currX + veloX + rectWidth, currY + veloY + rectWidth);
		grabCut(currFrame, markers, rectangle, bg, fg, 5, GC_INIT_WITH_RECT);
		
		int count = 0;
		int sumw = 0, sumh = 0;
		for (int h = 0; h < currFrame.rows; h++) {
			for (int w = 0; w < currFrame.cols; w++) {
				if (markers.at<uchar>(h, w) == GC_PR_FGD || markers.at<uchar>(h, w) == GC_FGD) {
					count++;
					sumw += w;
					sumh += h;
				}
			}
		}

		currX = sumw / count - rectWidth / 2;
		currY = sumh / count - rectWidth / 2;
		veloX = currX - preX;
		veloY = currY - preY;
		compare(markers, GC_PR_FGD, markers, CMP_EQ);
		Mat frame(currFrame.size(), CV_8UC3, Scalar(255, 255, 255));
		currFrame.copyTo(frame, markers);
		
		imshow("rec", frame);
		waitKey(5);
		preX = currX;
		preY = currY;


	}
}