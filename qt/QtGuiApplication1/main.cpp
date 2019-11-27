#include "QtGuiApplication1.h"
#include <QtWidgets/QApplication>
#include <QThread>
#include <QMainWindow>


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();




    return a.exec();
}

