#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "src/myqrcodeheader.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    MyQRcode qrcode("https://www.baidu.com",QSize(300,300));
    QPixmap pix;
    qrcode.QRCodeGenerate(pix);
    ui->label->setPixmap(pix);

    QZXing zxing;
    QString str=zxing.decodeImage(QImage(pix.toImage()));
    ui->label_2->setText(str);
}

MainWindow::~MainWindow()
{
    delete ui;
}
