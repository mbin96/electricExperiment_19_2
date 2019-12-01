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
    cv::Mat reset_shape; //= cv::Mat::zeros(1, 1, CV_8UC1);
    int numLandmarks;
    void setKemonoEar(cv::Mat& _inOutAlphaImg, cv::Point& kemonoLocation, int AlphaImgFaceWidth);
    std::string item_name;
    int kemonoFaceWidth = 180;
    cv::Mat kemonoEar ;
    cv::Mat kemonoNose;

    void setKemonoNose(cv::Mat& _inOutAlphaImg, cv::Point& kemonoLocation);
    BlendItem blend;
    Ui::QtGuiApplication2Class ui;
    cv::Point faceEyeCenter;
    cv::Point faceBottom;
    // OpenCV class
    cv::VideoCapture capture;
    cv::Mat original;

    // QImage class
    QImage qimgOriginal;

    // QTimer class - to give time to respond to UI
    QTimer* tmrTimer;

public slots:

    void processFrameAndUpdateGUI();
    void clearEarButton();
    void clearNoseButton();
    void openVideo();
    void openCam();

    void catNoseButton();
    void catEarButton();
    void rabbitEarButton();
    void On_Clicked_OpenEarImgFiles();
    void On_Clicked_OpenNoseImgFiles();
    //bool eventFilter(QObject* object, QEvent* event);
};
