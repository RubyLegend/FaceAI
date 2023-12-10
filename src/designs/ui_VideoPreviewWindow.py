# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'VideoPreviewWindowUpLJuo.ui'
##
## Created by: Qt User Interface Compiler version 5.15.11
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *  # type: ignore
from PySide2.QtGui import *  # type: ignore
from PySide2.QtWidgets import *  # type: ignore


class Ui_VideoPreviewWindow(QWidget):
    def setupUi(self, VideoPreviewWindow):
        if not VideoPreviewWindow.objectName():
            VideoPreviewWindow.setObjectName(u"VideoPreviewWindow")
        VideoPreviewWindow.resize(800, 600)
        VideoPreviewWindow.setMinimumSize(QSize(800, 0))
        VideoPreviewWindow.setMaximumSize(QSize(800, 600))
        self.verticalLayout = QVBoxLayout(VideoPreviewWindow)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.ImageFeed = QLabel(VideoPreviewWindow)
        self.ImageFeed.setObjectName(u"ImageFeed")
        sizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ImageFeed.sizePolicy().hasHeightForWidth())
        self.ImageFeed.setSizePolicy(sizePolicy)
        self.ImageFeed.setScaledContents(True)
        self.ImageFeed.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.ImageFeed)


        self.retranslateUi(VideoPreviewWindow)

        QMetaObject.connectSlotsByName(VideoPreviewWindow)
    # setupUi

    def retranslateUi(self, VideoPreviewWindow):
        VideoPreviewWindow.setWindowTitle(QCoreApplication.translate("VideoPreviewWindow", u"Video Preview", None))
        self.ImageFeed.setText(QCoreApplication.translate("VideoPreviewWindow", u"TextLabel", None))
    # retranslateUi

