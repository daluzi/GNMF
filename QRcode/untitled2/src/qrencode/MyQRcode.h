#ifndef MYQRCODE_H
#define MYQRCODE_H

#include <QDebug>
#include <QIcon>
#include <QWidget>
#include <QPainter>
#include <QSpacerItem>
#include <QByteArray>
#include <QFile>
#include "qrencode/qrencode.h"

/*******************************************************************
 *
 *  Autor:   kingderzhang
 *  QQ:      543985125
 *  Time :   2018.3.7
 *  Version: 1.0.0
 *
 *  函数：
 *    MyQRcode(const QByteArray text,const QSize size)；构造函数，参数信息为二维码数据，二维码的大小
 *
 *    setQRcodeMargin(const int &margin=10)；设置图片中二维码距离四周边缘的距离
 *
 *    setQRcodeIcon(const QPixmap &icon, qreal percent=0.23)；设置二维码中心图片，参数为图片内容，图片在整个二维码高度的比例
 *
 *    setQRcodeInfo(QByteArray str);设置二维码的内容
 *
 *    setQRcodeSize(QSize size)；设置二维码的大小
 *
 *    QRCodeGenerate(QPixmap &pix);生成二维码并保存在pixmap中
 *
 *
 * 调用举例：
 *
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
 *
 *
 *
*******************************************************************/


class MyQRcode{

public:
    MyQRcode(const QByteArray text,const QSize size){
        if (text.length () == 0) {
            m_textInfo = QByteArray("QRCODE No data!");
        }
        else {
            m_textInfo = text;
        }
        m_margin = 10;
        m_foreground = QColor("black");
        m_background = QColor("white");
        m_casesen = true;
        m_mode = MODE_8;
        m_level = LEVEL_Q;
        m_percent = 0.23;
        m_size=size;
    }

    void setQRcodeMargin(const int &margin=10)
    {
        if(m_margin!=margin)
        {
         m_margin=margin;
        }
        else
            return;
    }

    void setQRcodeIcon(const QPixmap &icon, qreal percent=0.23)
    {
        m_percent = percent < 0.5 ? percent : 0.3;
        m_icon=icon;
    }

    void setQRcodeInfo(QByteArray str)
    {
        if(m_textInfo!=str)
        {
            m_textInfo=str;
        }
        else {
            return;
        }
    }

    void setQRcodeSize(QSize size)
    {
        if(m_size!=size)
        {
          m_size=size;
        }
        else {
            return;
        }
    }

    void QRCodeGenerate(QPixmap &pix)
    {
        if(0==pix.width())
        {
            qDebug()<<"MyQRcode.h: QPixmap need set size.default set size(300,300)";
            QPixmap p(QSize(300,300));
            pix=p;
        }
        pix.fill();
        QPainter painter(&pix);
        QRcode *qrcode = QRcode_encodeString(m_textInfo.data (), 7, (QRecLevel)m_level, (QRencodeMode)m_mode, m_casesen ? 1 : 0);
        if(0 != qrcode)
        {
            unsigned char *point = qrcode->data;
            painter.setPen(Qt::NoPen);
            painter.setBrush(m_background);
            painter.drawRect(0, 0,m_size.width(), m_size.height());
            double scale = (m_size.width() - 2.0 * m_margin) / qrcode->width;
            painter.setBrush(m_foreground);

            for (int y = 0; y < qrcode->width; y ++)
            {
                for (int x = 0; x < qrcode->width; x ++)
                {
                    if (*point & 1)
                    {
                        QRectF r(m_margin + x * scale, m_margin + y * scale, scale, scale);
                        painter.drawRects(&r, 1);
                    }
                    point ++;
                }
            }

            point = NULL;
            QRcode_free(qrcode);

            //画中心图标
            if(!m_icon.isNull())
            {
                painter.setBrush(m_background);
                double icon_width = (m_size.width() - 2.0 * m_margin) * m_percent;
                double icon_height = icon_width;
                double wrap_x = (m_size.width() - icon_width) / 2.0;
                double wrap_y = (m_size.width() - icon_height) / 2.0;
                QRectF wrap(wrap_x - 5, wrap_y - 5, icon_width + 10, icon_height + 10);
                painter.drawRoundRect(wrap, 50, 50);

                QPixmap image(m_icon);

                QRectF target(wrap_x, wrap_y, icon_width, icon_height);
                QRectF source(0, 0, image.width(), image.height());
                painter.drawPixmap (target, image, source);
            }

        }
        qrcode = NULL;
    }

    enum QR_MODE {
        MODE_NUL = QR_MODE_NUL,
        MODE_NUM = QR_MODE_NUM,
        MODE_AN = QR_MODE_AN,
        MODE_8 = QR_MODE_8,
        MODE_KANJI = QR_MODE_KANJI,
        MODE_STRUCTURE = QR_MODE_STRUCTURE,
        MODE_ECI = QR_MODE_ECI,
        MODE_FNC1FIRST = QR_MODE_FNC1FIRST,
        MODE_FNC1SECOND = QR_MODE_FNC1SECOND
    };

    enum QR_LEVEL {
        LEVEL_L = QR_ECLEVEL_L,
        LEVEL_M = QR_ECLEVEL_M,
        LEVEL_Q = QR_ECLEVEL_Q,
        LEVEL_H = QR_ECLEVEL_H
    };
protected:
    void caseSensitive(bool flag)
    {
        m_casesen=flag;
    }

private:
    bool m_casesen;
    int m_margin=10;//二维码边缘间隙
    QSize m_size;   //设置二维码的大小
    QPixmap m_icon;   //设置中心图片
    qreal m_percent;
    QByteArray m_textInfo;
    QColor m_foreground;
    QColor m_background;
    QR_MODE m_mode;
    QR_LEVEL m_level;
};


#endif
