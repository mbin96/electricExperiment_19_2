#include "QtGuiApplication2.h"

QtGuiApplication2::QtGuiApplication2(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    capture.open("movie.mp4");
    // Check if sucessful
    if (!capture.isOpened() == true) {
        //ui->txtXYRadius->appendPlainText("error: camera error!");
        return;
    }
    capture >> original;
    tmrTimer = new QTimer(this);
    connect(tmrTimer, SIGNAL(timeout()), this, SLOT(processFrameAndUpdateGUI()));
    tmrTimer->start(30);
}


void QtGuiApplication2::processFrameAndUpdateGUI() {
    capture >> original;

    if (original.empty() == true)
        return;
    //cv::GaussianBlur(processed, processed, cv::Size(9, 9), 1.5);
    //imshow("1",original);
    // OpenCV to QImage datatype to display on labels
    cv::cvtColor(original, original, cv::COLOR_BGR2RGB);
    QImage qimgOriginal((uchar*)original.data, original.cols, original.rows, original.step, QImage::Format_RGB888); // for color images
    ui.label-> setScaledContents(1);
    ui.label->setPixmap(QPixmap::fromImage(qimgOriginal));
}