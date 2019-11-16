#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
void main()
{
	Mat image = imread("bird.jpg");
	Mat imgMask = imread("bird_mask.bmp");
	Rect rectangle(0, 0, image.rows - 1, image.cols - 1);
	Mat bg, fg;
	cv::Mat1b markers(image.rows, image.cols);

	cv::Mat1b markers_(image.rows, image.cols);
	for (int h = 0; h < image.rows; h++) {
		for (int w = 0; w < image.cols; w++) {
			markers.at<uchar>(h, w) = GC_PR_BGD;
			if (imgMask.at<Vec3b>(h, w)[2] == 255) {
				markers.at<uchar>(h, w) = GC_PR_FGD;

			}
			if (imgMask.at<Vec3b>(h, w)[0] == 255) {
				markers.at<uchar>(h, w) = GC_BGD;
			}
		}
	}

	grabCut(image, markers, rectangle, bg, fg, 5, GC_INIT_WITH_MASK);

	Mat foreground(image.size(), CV_8UC3, Scalar(255, 255, 255));
	compare(markers, GC_PR_FGD, markers, CMP_EQ);
	image.copyTo(foreground, markers);

	imshow("Foreground", foreground);
	waitKey(5000);
}