
 *  Autor:   kingderzhang
 *  QQ:      543985125
 *  Time :   2018.3.7
 *  Version: 1.0.0

一、本文件结构
    src
     |____qrencode
     |
     |____zxing
     |
     |____QZxing .pro


二、配置说明：
  1、首先将本项目所有文件拷贝至你的工程项目下，使整个项目结构如下所示
     your project.pro
     |
     |
    src
     |____qrencode
     |
     |____zxing
     |
     |____QZxing .pro

  2、在你的项目的工程文件中需要添加这一句话，包含当前新添加的项目
     include(./src/QZXing.pri)

  3、基本的配置就已经完成了，保存项目文件。

三、如何使用：
  1、首先在需要的文件中包含所有需要使用到的头文件
      #include "src/myqrcodeheader.h"
    备注：只需要包含此头文件即可进行二维码的生成和（一、二维码）解析。

  2、生成二维码
     生成二维码需要使用的类是 MyQRcode 类
     此类中可能会需要使用的函数如下：
 *  函数：
 *    MyQRcode::MyQRcode(const QByteArray text,const QSize size)；构造函数，参数信息为二维码数据，二维码的大小
 *
 *    MyQRcode::setQRcodeMargin(const int &margin=10)；设置图片中二维码距离四周边缘的距离
 *
 *    MyQRcode::setQRcodeIcon(const QPixmap &icon, qreal percent=0.23)；设置二维码中心图片，参数为图片内容，图片在整个二维码高度的比例
 *
 *    MyQRcode::setQRcodeInfo(QByteArray str);设置二维码的内容
 *
 *    MyQRcode::setQRcodeSize(QSize size)；设置二维码的大小
 *
 *    MyQRcode::QRCodeGenerate(QPixmap &pix);生成二维码并保存在pixmap中

 调用举例：

 *   1、构造QRcode对象
 *   MyQRcode qrcode("https://www.baidu.com",QSize(300,300));
 *
 *   2、设置中心图片
 *   qrcode.setQRcodeIcon(QPixmap(":/new/prefix1/logo.png"),0.2);
 *
 *   3、设置二维码信息
 *   qrcode.setQRcodeInfo("你好，老哥！！！");
 *
 *   4、设置边缘距离
 *   qrcode.setQRcodeMargin(10);
 *
 *   5、定义一个pixmap对象用于保存生成的二维码图像
 *   QPixmap pix;
 *
 *   6、传入pixmap对象并生成二维码
 *   qrcode.QRCodeGenerate(pix);
 *
 *   7、通过pixmap对二维码进行使用
 *   ui->label->setPixmap(pix);

 3、解析二维码
    解析二维码需要使用的类是 QZXing 类

    解析二维码调用接口非常简单
    
    可能需要使用的函数：
    QString  QZXing::decodeImage(QImage &img);//输入解析图像返回信息内容
    QString  QZXing::decodeImageFromFile(QString &path);//根据图像路径返回信息内容

    备注：如果解析失败，返回空字符串

  调用举例：
    1、定义一个识别对象
    QZXing zxing;

    2、将图像进行识别并返回内容
    QString str=zxing.decodeImage(QImage(pix.toImage()));
