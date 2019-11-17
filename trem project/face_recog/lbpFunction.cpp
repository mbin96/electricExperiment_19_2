#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "lbpFunction.hpp"

//lbp 로드해 오기
cv::Mat loadlbp(std::string lbpFile) {

    cv::Mat outImg = cv::Mat::zeros(LBP_INPUT_SIZE * LBP_DIMEN_SIZE, LBP_INPUT_SIZE, CV_8UC1);

    int tmp;
    //open weight
    std::ifstream in(lbpFile);

    //
    for (int h = 0; h < LBP_INPUT_SIZE * LBP_DIMEN_SIZE; h++) {
        for (int w = 0; w < LBP_INPUT_SIZE; w++) {
            in >> tmp;
            outImg.at<uchar>(h, w) = (uchar)tmp;
        }
    }
    return outImg;
}

int savelbp(cv::Mat lbpImg, std::string lbpFile) {

    //open weight
    std::ofstream out(lbpFile);

    for (int h = 0; h < LBP_INPUT_SIZE * LBP_DIMEN_SIZE; h++) {
        for (int w = 0; w < LBP_INPUT_SIZE; w++) {
            out << (uint)lbpImg.at<uchar>(h, w) << std::endl;
        }
    }

    return 0;
}



// LBP_INPUT_SIZE 만큼의 정사각형 이미지를 만들어줌(컬러)
cv::Mat lbpCut(cv::Mat origImg, int x, int y) {

    //들어온 좌표를 가운데로 하기위해 빼주기
    int startX = x - LBP_INPUT_SIZE / 2;
    int startY = y - LBP_INPUT_SIZE / 2;

    int h, w;

    //outputimg
    cv::Mat outImg = cv::Mat::zeros(LBP_INPUT_SIZE, LBP_INPUT_SIZE, CV_8UC3);

    for (h = 0; h < LBP_INPUT_SIZE; h++) {
        for (w = 0; w < LBP_INPUT_SIZE; w++) {
            outImg.at<cv::Vec3b>(h, w) = origImg.at<cv::Vec3b>(startY + h, startX + w);
        }
    }
    return outImg;
}

//컬러이미지를 받아서 흑백 LBP 이미지를 만들어줌.
cv::Mat lbpImg(cv::Mat origImg) {
    int i, j, h, w, ii, jj;
    float max = INT_MIN;

    cv::Mat origImgGray = cv::Mat::zeros(LBP_INPUT_SIZE, LBP_INPUT_SIZE, CV_8UC1);
    for (i = 0; i < LBP_INPUT_SIZE; i++) {
        //X
        for (j = 0; j < LBP_INPUT_SIZE; j++) {
            origImgGray.at<uchar>(i, j) = (origImg.at<cv::Vec3b>(i, j)[0] + origImg.at<cv::Vec3b>(i, j)[1] + origImg.at<cv::Vec3b>(i, j)[2]) / 3;
        }
    }

    cv::Mat outImg = cv::Mat::zeros(LBP_INPUT_SIZE, LBP_INPUT_SIZE, CV_8UC1);
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


//커다란 디멘젼으로 만들어줌
cv::Mat lbpDimen(cv::Mat origImg[LBP_DIMEN_SIZE]) {
    int i, h, w;
    cv::Mat outImg = cv::Mat::zeros(LBP_INPUT_SIZE * LBP_DIMEN_SIZE, LBP_INPUT_SIZE, CV_8UC1);

    for (i = 0; i < LBP_DIMEN_SIZE; i++) {
        for (h = 0; h < LBP_INPUT_SIZE; h++) {
            for (w = 0; w < LBP_INPUT_SIZE; w++) {
                outImg.at<uchar>(i * LBP_INPUT_SIZE + h, w) = (origImg[i]).at<uchar>(h,w);
            }
        }



        outImg = origImg[i];
    }
}


int lbpComp(cv::Mat ref, cv::Mat tar, int dimension) {
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
