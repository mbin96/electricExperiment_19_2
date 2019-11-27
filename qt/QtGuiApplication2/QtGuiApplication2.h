#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_QtGuiApplication2.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <QTimer>
class QtGuiApplication2 : public QMainWindow
{
    Q_OBJECT

public:
    QtGuiApplication2(QWidget *parent = Q_NULLPTR);

private:
    Ui::QtGuiApplication2Class ui;

    // OpenCV class
    cv::VideoCapture capture;
    cv::Mat original;
    cv::Mat processed;
    // OpenCV variables
    std::vector<cv::Vec3f> vecCircles;
    std::vector<cv::Vec3f>::iterator itrCircles;
    double param1, param2;
    int minRadius, maxRadius;

    // QImage class
    QImage qimgOriginal;
    QImage qimgProcessed;

    // QTimer class - to give time to respond to UI
    QTimer* tmrTimer;

public slots:
    void processFrameAndUpdateGUI();
    //bool eventFilter(QObject* object, QEvent* event);
};
