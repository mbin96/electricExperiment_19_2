#include "QtGuiApplication2.h"
#include "qfile.h"
#include <QFileSystemWatcher.h>
#include <QFileDialog>

//#include "trackImg.cpp"
#include "ldmarkmodel.h"

ldmarkmodel modelt;


QtGuiApplication2::QtGuiApplication2(QWidget *parent)

    : QMainWindow(parent)
{

    ui.setupUi(this);
    


    connect(ui.earButtonCat, SIGNAL(clicked()), this, SLOT(catEarButton()));
    connect(ui.earButtonBunny , SIGNAL(clicked()), this, SLOT(rabbitEarButton()));
    connect(ui.earButtonFile, SIGNAL(clicked()), this, SLOT(On_Clicked_OpenEarImgFiles()));
    connect(ui.earButtonClear, SIGNAL(clicked()), this, SLOT(clearEarButton()));

    connect(ui.noseButtonCat, SIGNAL(clicked()), this, SLOT(catNoseButton()));
    connect(ui.noseButtonFile, SIGNAL(clicked()), this, SLOT(On_Clicked_OpenNoseImgFiles()));
    connect(ui.noseButtonClear, SIGNAL(clicked()), this, SLOT(clearNoseButton()));

    connect(ui.openButtonCam, SIGNAL(clicked()), this, SLOT(openCam()));
    connect(ui.openButtonVideo, SIGNAL(clicked()), this, SLOT(openVideo()));
    load_ldmarkmodel("roboman-landmark-model.bin", modelt);
    
    tmrTimer = new QTimer(this);
    connect(tmrTimer, SIGNAL(timeout()), this, SLOT(processFrameAndUpdateGUI()));
    tmrTimer->start(30);
    //ui.label->setText(QString("camera reading"));
}

void QtGuiApplication2::openVideo() {
    QString strFilter = "mp4 file (*.mp4) ;; webM file (*.webm) ;; mkv file (*.mkv) ;; All files (*.*)";
    QString strFileName = QFileDialog::getOpenFileName(this, "Open a image file", QDir::currentPath(), strFilter);
    std::string file = strFileName.toStdString();
    
    capture.open(file);
    // Check if sucessful
    if (!capture.isOpened() == true) {
        ui.label->setText("error: camera error!");
        return;
    }

}

void QtGuiApplication2::openCam() {
    capture.open(0);
    // Check if sucessful
    if (!capture.isOpened() == true) {
        ui.label->setText("error: camera error!");
        return;
    }

}


void QtGuiApplication2::clearEarButton() {
    kemonoEar = reset_shape;
}

void QtGuiApplication2::clearNoseButton() {
    kemonoNose = reset_shape;
}

void QtGuiApplication2::On_Clicked_OpenEarImgFiles()
{
    QString strFilter = "png file (*.png) ;; webP file (*.webp) ;; All files (*.*)";
    QString strFileName = QFileDialog::getOpenFileName(this, "Open a image file", QDir::currentPath(), strFilter);
    std::string file = strFileName.toStdString();
    kemonoEar = imread(file, cv::IMREAD_UNCHANGED);
    kemonoFaceWidth = 2 * kemonoEar.cols / 3 + 1;
}
void QtGuiApplication2::On_Clicked_OpenNoseImgFiles()
{
    QString strFilter = "png file (*.png) ;; webP file (*.webp) ;; All files (*.*)";
    QString strFileName = QFileDialog::getOpenFileName(this, "Open a image file", QDir::currentPath(), strFilter);
    std::string file = strFileName.toStdString();
    kemonoNose = imread(file, cv::IMREAD_UNCHANGED);
}


void QtGuiApplication2::catEarButton() {
    kemonoEar = imread("cat_ear.png", cv::IMREAD_UNCHANGED);
    ui.label->setText(QString("item changed"));
    kemonoFaceWidth = 150;
}
void QtGuiApplication2::rabbitEarButton() {
    kemonoEar = imread("rabbit_ear_1.png", cv::IMREAD_UNCHANGED);
    ui.label->setText(QString("item changed"));
    kemonoFaceWidth = 180;
}
void QtGuiApplication2::catNoseButton() {
    kemonoNose = imread("cat_nose.png", cv::IMREAD_UNCHANGED);
    ui.label->setText(QString("item changed"));
}

void QtGuiApplication2::processFrameAndUpdateGUI() {
    capture >> original;
   
    if (original.empty() == true)
        return;

    
    //Mat frame;
    Mat alphaItem, image;
    
    Point itemPoint;
    original.copyTo(image);

   

    modelt.track(image, current_shape);
    modelt.EstimateHeadPose(current_shape, eav);
    //modelt.drawPose(image, current_shape, 50);
    numLandmarks = current_shape.cols / 2;
    
    if (numLandmarks >= 68) {
        blend.setBg(image);
        if (kemonoEar.rows > 10) {

            kemonoEar.copyTo(alphaItem);
            setKemonoEar(alphaItem, itemPoint, kemonoFaceWidth);

            
                blend.blendItem(alphaItem, itemPoint);
                ui.label->setText(QString("face detected"));
        }
        if (kemonoNose.rows > 10) {
            kemonoNose.copyTo(alphaItem);
            setKemonoNose(alphaItem, itemPoint);
            
                blend.blendItem(alphaItem, itemPoint);
                ui.label->setText(QString("face detected"));
        }
        if (alphaItem.cols > 2 * image.rows / 3 || alphaItem.cols < image.rows / 5) {
            current_shape = reset_shape;
        }
        blend.getBlended(image);
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    QImage qimgOriginal((uchar*)image.data, image.cols, image.rows, image.step, QImage::Format_RGB888); // for color images
    ui.camViewer-> setScaledContents(1);
    ui.camViewer->setPixmap(QPixmap::fromImage(qimgOriginal));
}



void QtGuiApplication2::setKemonoNose(cv::Mat& _inOutAlphaImg, cv::Point& kemonoLocation) {

    cv::Point facebrowLeft = cv::Point(current_shape.at<float>(17), current_shape.at<float>(17 + numLandmarks));
    cv::Point facebrowRight = cv::Point(current_shape.at<float>(26), current_shape.at<float>(26 + numLandmarks));
    faceEyeCenter = cv::Point((current_shape.at<float>(21) + current_shape.at<float>(22)) / 2, (current_shape.at<float>(21 + numLandmarks) + current_shape.at<float>(22 + numLandmarks)) / 2);
    faceBottom = cv::Point(current_shape.at<float>(8), current_shape.at<float>(8 + numLandmarks));

    int itemSize = _inOutAlphaImg.rows* (0.8 * sqrt(pow(faceBottom.x - faceEyeCenter.x, 2) + pow(faceBottom.y - faceEyeCenter.y, 2))) / _inOutAlphaImg.cols;
    cv::resize(_inOutAlphaImg, _inOutAlphaImg, cv::Size(itemSize*1.5, itemSize * 1.5));//(cos(eav[0]*3.14 / 180))));
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point(_inOutAlphaImg.rows / 2, _inOutAlphaImg.cols / 2), -eav[2], 1.0);
    cv::warpAffine(_inOutAlphaImg, _inOutAlphaImg, rot, cv::Size(_inOutAlphaImg.cols, _inOutAlphaImg.rows), 1, cv::BORDER_TRANSPARENT);
    cv::Point noseTop = cv::Point(current_shape.at<float>(30), current_shape.at<float>(30 + numLandmarks));
    kemonoLocation = noseTop;
}

void QtGuiApplication2::setKemonoEar(cv::Mat& _inOutAlphaImg, cv::Point& kemonoLocation, int AlphaImgFaceWidth) {

    cv::Point facebrowLeft = cv::Point(current_shape.at<float>(17), current_shape.at<float>(17 + numLandmarks));
    cv::Point facebrowRight = cv::Point(current_shape.at<float>(26), current_shape.at<float>(26 + numLandmarks));
    faceEyeCenter = cv::Point((current_shape.at<float>(21) + current_shape.at<float>(22)) / 2, (current_shape.at<float>(21 + numLandmarks) + current_shape.at<float>(22 + numLandmarks)) / 2);
    faceBottom = cv::Point(current_shape.at<float>(8), current_shape.at<float>(8 + numLandmarks));

    cv::Point noseTop = cv::Point(current_shape.at<float>(30), current_shape.at<float>(30 + numLandmarks));
    float faceHigh = sqrt(pow(faceEyeCenter.x - noseTop.x, 2) + pow(faceEyeCenter.y - noseTop.y, 2));
    
    float faceCenterX = (float)(current_shape.at<float>(2) / 2 + current_shape.at<float>(14) / 2);
    cv::Point faceTop = cv::Point(faceEyeCenter.x - 0.7 * (faceBottom.x - faceCenterX), faceEyeCenter.y - faceHigh * pow(cos(eav[2] * 3.14 / 180), 2));
    float faceWidth = sqrt(pow(facebrowLeft.x - facebrowRight.x, 2) + pow(facebrowLeft.y - facebrowRight.y, 2));
    
    
    cv::resize(_inOutAlphaImg, _inOutAlphaImg, cv::Size(_inOutAlphaImg.cols * (faceWidth / AlphaImgFaceWidth), _inOutAlphaImg.rows * (0.8 * sqrt(pow(faceBottom.x - faceEyeCenter.x, 2) + pow(faceBottom.y - faceEyeCenter.y, 2))) / AlphaImgFaceWidth));//(cos(eav[0]*3.14 / 180))));
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point(_inOutAlphaImg.rows / 2, _inOutAlphaImg.cols / 2), -eav[2], 1.0);
    cv::warpAffine(_inOutAlphaImg, _inOutAlphaImg, rot, cv::Size(_inOutAlphaImg.cols, _inOutAlphaImg.rows), 1, cv::BORDER_TRANSPARENT);

    
    kemonoLocation = faceTop;
}

