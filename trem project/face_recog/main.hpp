#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/ml/ml.hpp>
#include "ldmarkmodel.h"


#define FACE_IMG_SIZE 128

std::vector<cv::Rect> doCascacade(cv::Mat input);
cv::Mat* getFaceImg(cv::Mat input, int imgSize, std::vector<cv::Rect> faces);