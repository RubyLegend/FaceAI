# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ErrorWindowvpvLqv.ui'
##
## Created by: Qt User Interface Compiler version 5.15.11
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *  # type: ignore
from PySide2.QtGui import *  # type: ignore
from PySide2.QtWidgets import *  # type: ignore


class Ui_ErrorWindow(QDialog):
    def setupUi(self, ErrorWindow):
        if not ErrorWindow.objectName():
            ErrorWindow.setObjectName(u"ErrorWindow")
        ErrorWindow.resize(600, 300)
        ErrorWindow.setMinimumSize(QSize(600, 300))
        self.verticalLayout_2 = QVBoxLayout(ErrorWindow)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label = QLabel(ErrorWindow)
        self.label.setObjectName(u"label")
        self.label.setMinimumSize(QSize(0, 30))
        self.label.setMaximumSize(QSize(16777215, 30))
        font = QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.label)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_2)

        self.ErrorMessage = QLabel(ErrorWindow)
        self.ErrorMessage.setObjectName(u"ErrorMessage")
        font1 = QFont()
        font1.setPointSize(12)
        self.ErrorMessage.setFont(font1)
        self.ErrorMessage.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.ErrorMessage)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setSizeConstraint(QLayout.SetMinimumSize)
        self.OkButton = QPushButton(ErrorWindow)
        self.OkButton.setObjectName(u"OkButton")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.OkButton.sizePolicy().hasHeightForWidth())
        self.OkButton.setSizePolicy(sizePolicy)
        self.OkButton.setMinimumSize(QSize(0, 30))
        self.OkButton.setMaximumSize(QSize(16777215, 30))

        self.horizontalLayout.addWidget(self.OkButton)


        self.verticalLayout.addLayout(self.horizontalLayout)


        self.verticalLayout_2.addLayout(self.verticalLayout)


        self.retranslateUi(ErrorWindow)
        self.OkButton.clicked.connect(ErrorWindow.hide)

        QMetaObject.connectSlotsByName(ErrorWindow)
    # setupUi

    def retranslateUi(self, ErrorWindow):
        ErrorWindow.setWindowTitle(QCoreApplication.translate("ErrorWindow", u"Dialog", None))
        self.label.setText(QCoreApplication.translate("ErrorWindow", u"Error", None))
        self.ErrorMessage.setText(QCoreApplication.translate("ErrorWindow", u"ErrorMessage", None))
        self.OkButton.setText(QCoreApplication.translate("ErrorWindow", u"OK", None))
    # retranslateUi

