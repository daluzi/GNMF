
 *  Autor:   kingderzhang
 *  QQ:      543985125
 *  Time :   2018.3.7
 *  Version: 1.0.0

һ�����ļ��ṹ
    src
     |____qrencode
     |
     |____zxing
     |
     |____QZxing .pro


��������˵����
  1�����Ƚ�����Ŀ�����ļ���������Ĺ�����Ŀ�£�ʹ������Ŀ�ṹ������ʾ
     your project.pro
     |
     |
    src
     |____qrencode
     |
     |____zxing
     |
     |____QZxing .pro

  2���������Ŀ�Ĺ����ļ�����Ҫ�����һ�仰��������ǰ����ӵ���Ŀ
     include(./src/QZXing.pri)

  3�����������þ��Ѿ�����ˣ�������Ŀ�ļ���

�������ʹ�ã�
  1����������Ҫ���ļ��а���������Ҫʹ�õ���ͷ�ļ�
      #include "src/myqrcodeheader.h"
    ��ע��ֻ��Ҫ������ͷ�ļ����ɽ��ж�ά������ɺͣ�һ����ά�룩������

  2�����ɶ�ά��
     ���ɶ�ά����Ҫʹ�õ����� MyQRcode ��
     �����п��ܻ���Ҫʹ�õĺ������£�
 *  ������
 *    MyQRcode::MyQRcode(const QByteArray text,const QSize size)�����캯����������ϢΪ��ά�����ݣ���ά��Ĵ�С
 *
 *    MyQRcode::setQRcodeMargin(const int &margin=10)������ͼƬ�ж�ά��������ܱ�Ե�ľ���
 *
 *    MyQRcode::setQRcodeIcon(const QPixmap &icon, qreal percent=0.23)�����ö�ά������ͼƬ������ΪͼƬ���ݣ�ͼƬ��������ά��߶ȵı���
 *
 *    MyQRcode::setQRcodeInfo(QByteArray str);���ö�ά�������
 *
 *    MyQRcode::setQRcodeSize(QSize size)�����ö�ά��Ĵ�С
 *
 *    MyQRcode::QRCodeGenerate(QPixmap &pix);���ɶ�ά�벢������pixmap��

 ���þ�����

 *   1������QRcode����
 *   MyQRcode qrcode("https://www.baidu.com",QSize(300,300));
 *
 *   2����������ͼƬ
 *   qrcode.setQRcodeIcon(QPixmap(":/new/prefix1/logo.png"),0.2);
 *
 *   3�����ö�ά����Ϣ
 *   qrcode.setQRcodeInfo("��ã��ϸ磡����");
 *
 *   4�����ñ�Ե����
 *   qrcode.setQRcodeMargin(10);
 *
 *   5������һ��pixmap�������ڱ������ɵĶ�ά��ͼ��
 *   QPixmap pix;
 *
 *   6������pixmap�������ɶ�ά��
 *   qrcode.QRCodeGenerate(pix);
 *
 *   7��ͨ��pixmap�Զ�ά�����ʹ��
 *   ui->label->setPixmap(pix);

 3��������ά��
    ������ά����Ҫʹ�õ����� QZXing ��

    ������ά����ýӿڷǳ���
    
    ������Ҫʹ�õĺ�����
    QString  QZXing::decodeImage(QImage &img);//�������ͼ�񷵻���Ϣ����
    QString  QZXing::decodeImageFromFile(QString &path);//����ͼ��·��������Ϣ����

    ��ע���������ʧ�ܣ����ؿ��ַ���

  ���þ�����
    1������һ��ʶ�����
    QZXing zxing;

    2����ͼ�����ʶ�𲢷�������
    QString str=zxing.decodeImage(QImage(pix.toImage()));
