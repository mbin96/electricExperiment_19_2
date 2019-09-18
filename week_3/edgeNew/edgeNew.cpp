#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#pragma warning(disable:4996)

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

int main()
{
	Mat imgColor = imread("test.jpg", cv::IMREAD_COLOR); // image를 Mat 객체로 읽어오기	
	int width = imgColor.cols;
	int height = imgColor.rows;

	Mat imgGray(height, width, CV_8UC1);
	Mat imgEdge(height, width, CV_8UC1);

	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++)
		{
			imgGray.at<uchar>(i, j) = (imgColor.at<Vec3b>(i, j)[2] + imgColor.at<Vec3b>(i, j)[1] + imgColor.at<Vec3b>(i, j)[0]) / 3;
		}
	}

	int sumX, sumY;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
		{
			sumX = sumY = 0;
			for (int h = 0; h < 3; h++) {
				for (int w = 0; w < 3; w++) {
					sumX += imgGray.at<uchar>(i, j) * filterX[h * 3 + w];
					sumY += imgGray.at<uchar>(i, j) * filterY[h * 3 + w]/3;
				}
			}
		}
	}
	sumX = 0;

	imshow("result",imgEdge);
	waitKey(5000);
}
