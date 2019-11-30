#pragma once


#include <QtWidgets/QMainWindow>
#include "ui_QtGuiApplication2.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <QTimer>
#include "blendItem.h"
class QtGuiApplication2 : public QMainWindow
{
    Q_OBJECT

public:
    QtGuiApplication2(QWidget *parent = Q_NULLPTR);

private:
    
    cv::Mat current_shape;
    cv::Vec3d eav;
    cv::Mat reset_shape = cv::Mat::zeros(1, 1, CV_8UC1);
    int numLandmarks;
    void setKemonoEar(cv::Mat& _inOutAlphaImg, cv::Point& kemonoLocation, int AlphaImgFaceWidth, int shapeBorder);
    std::string item_name;
    int kemonoFaceWidth = 180;
    cv::Mat kemonoEar = imread("rabbit_ear_1.png", cv::IMREAD_UNCHANGED);

    BlendItem blend;
    Ui::QtGuiApplication2Class ui;

    // OpenCV class
    cv::VideoCapture capture;
    cv::Mat original;

    // QImage class
    QImage qimgOriginal;

    // QTimer class - to give time to respond to UI
    QTimer* tmrTimer;

public slots:
    void catEarButton();
    void processFrameAndUpdateGUI();
    void rabbitEarButton();
    //bool eventFilter(QObject* object, QEvent* event);
};
