#include "QtGuiApplication1.h"
#include "ui_mainwindow.h"

#include <QtCore>



MainWindow::MainWindow(QWidget *parent): 
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    cv::VideoCapture capture("movie.mp4");
    // Check if sucessful
    if (!capture.isOpened() == true) {
        //ui->txtXYRadius->appendPlainText("error: camera error!");
        return;
    }
    tmrTimer = new QTimer(this);
    connect(tmrTimer, SIGNAL(timeout()), this, SLOT(processFrameAndUpdateGUI()));
}

MainWindow::~MainWindow()
{
    delete ui;
}
void MainWindow::processFrameAndUpdateGUI() {
    capture.read(original);

    if (original.empty() == true)
        return;
    cv::GaussianBlur(processed, processed, cv::Size(9, 9), 1.5);

    // OpenCV to QImage datatype to display on labels
    cv::cvtColor(original, original, cv::COLOR_BGR2RGB);
    QImage qimgOriginal((uchar*)original.data, original.cols, original.rows, original.step, QImage::Format_RGB888); // for color images
    ui->label->setPixmap(QPixmap::fromImage(qimgOriginal));
}



