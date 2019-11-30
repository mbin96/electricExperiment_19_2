#include "QtGuiApplication2.h"

//#include "trackImg.cpp"
#include "ldmarkmodel.h"
union item {
    std::string name;
    int headWidthSize;
    int type;
};

ldmarkmodel modelt;


QtGuiApplication2::QtGuiApplication2(QWidget *parent)

    : QMainWindow(parent)
{

    ui.setupUi(this);
    capture.open("face.mp4");
    // Check if sucessful
    if (!capture.isOpened() == true) {
        //ui->txtXYRadius->appendPlainText("error: camera error!");
        return;
    }

    connect(ui.pushButton, SIGNAL(clicked()), this, SLOT(catEarButton()));

    connect(ui.toolButton , SIGNAL(clicked()), this, SLOT(rabbitEarButton()));
    
    load_ldmarkmodel("roboman-landmark-model.bin", modelt);
    
    tmrTimer = new QTimer(this);
    connect(tmrTimer, SIGNAL(timeout()), this, SLOT(processFrameAndUpdateGUI()));
    tmrTimer->start(10);
    ui.label->setText(QString("camera reading"));
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

void QtGuiApplication2::processFrameAndUpdateGUI() {
    capture >> original;

    if (original.empty() == true)
        return;

    
    //Mat frame;
    Mat alphaItem, image;
    
    Point itemPoint;
    original.copyTo(image);
    kemonoEar.copyTo(alphaItem);


    modelt.track(image, current_shape);
    modelt.EstimateHeadPose(current_shape, eav);
    //modelt.drawPose(image, current_shape, 50);
    numLandmarks = current_shape.cols / 2;

    ui.label->setText(QString("face not detected"));
    if (numLandmarks>0) {
        ui.label->setText(QString("face detected"));
        setKemonoEar(alphaItem, itemPoint, 180, image.rows/14);
        blend.setBg(image);
        blend.blendItem(alphaItem, itemPoint);
        blend.getBlended(image);
    }
    //cv::GaussianBlur(processed, processed, cv::Size(9, 9), 1.5);
    //imshow("1",original);
    // OpenCV to QImage datatype to display on labels
    
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    QImage qimgOriginal((uchar*)image.data, image.cols, image.rows, image.step, QImage::Format_RGB888); // for color images
    ui.camViewer-> setScaledContents(1);
    ui.camViewer->setPixmap(QPixmap::fromImage(qimgOriginal));
}










void QtGuiApplication2::setKemonoEar(cv::Mat& _inOutAlphaImg, cv::Point& kemonoLocation, int AlphaImgFaceWidth, int shapeBorder) {

    cv::Point facebrowLeft = cv::Point(current_shape.at<float>(17), current_shape.at<float>(17 + numLandmarks));
    cv::Point facebrowRight = cv::Point(current_shape.at<float>(26), current_shape.at<float>(26 + numLandmarks));
    cv::Point faceEyeCenter = cv::Point((current_shape.at<float>(21) + current_shape.at<float>(22)) / 2, (current_shape.at<float>(21 + numLandmarks) + current_shape.at<float>(22 + numLandmarks)) / 2);
    cv::Point faceBottom = cv::Point(current_shape.at<float>(8), current_shape.at<float>(8 + numLandmarks));

    cv::Point noseTop = cv::Point(current_shape.at<float>(30), current_shape.at<float>(30 + numLandmarks));
    int faceHigh = sqrt(pow(faceEyeCenter.x - noseTop.x, 2) + pow(faceEyeCenter.y - noseTop.y, 2));
    if (faceHigh < shapeBorder) {
        current_shape = reset_shape;
    }
    int faceCenterX = current_shape.at<float>(2) / 2 + current_shape.at<float>(14) / 2;
    // y �����ִ°� �󸶳� �������� �����غ���
    cv::Point faceTop = faceEyeCenter - cv::Point(0.7 * (faceBottom.x - faceCenterX), faceHigh * pow(cos(eav[2] * 3.14 / 180), 2));
    //faceTop.x = 2 * faceCenter.x - faceBottom.x;
    float faceWidth = sqrt(pow(facebrowLeft.x - facebrowRight.x, 2) + pow(facebrowLeft.y - facebrowRight.y, 2));
    //faceTop.y = faceCenter.y - faceWidth / ((faceBottom.y - current_shape.at<float>(30 + numLandmarks))/ (faceBottom.y - faceCenter.y));
    cv::resize(_inOutAlphaImg, _inOutAlphaImg, cv::Size(_inOutAlphaImg.cols * (faceWidth / AlphaImgFaceWidth), _inOutAlphaImg.rows * (0.8 * sqrt(pow(faceBottom.x - faceEyeCenter.x, 2) + pow(faceBottom.y - faceEyeCenter.y, 2))) / AlphaImgFaceWidth));//(cos(eav[0]*3.14 / 180))));
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point(_inOutAlphaImg.rows / 2, _inOutAlphaImg.cols / 2), -eav[2], 1.0);
    cv::warpAffine(_inOutAlphaImg, _inOutAlphaImg, rot, cv::Size(_inOutAlphaImg.cols, _inOutAlphaImg.rows), 1, cv::BORDER_TRANSPARENT);

    /*
    for (int j = 0; j < numLandmarks; j++) {
        int x = current_shape.at<float>(j);
        int y = current_shape.at<float>(j + numLandmarks);
        std::stringstream ss;
        ss << j;
        cv::putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.3, cv::Scalar(0, 0, 255));

        //Mat outImg = cv::Mat::zeros(LBP_INPUT_SIZE, LBP_INPUT_SIZE, CV_8UC1);

        //lbpImg(lbpCut(Image, x, y)).copyTo(outImg);

        //cv::imshow("lbp" + std::to_string(j), lbpImg(lbpCut(Image, x, y)));


        //            cv::putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 0, 255));
        cv::circle(Image, cv::Point(x, y), 10, cv::Scalar(0, 0, 255), -1);
    }
    */
    kemonoLocation = faceTop;
}

