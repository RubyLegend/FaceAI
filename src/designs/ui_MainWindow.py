# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindowkYfegZ.ui'
##
## Created by: Qt User Interface Compiler version 5.15.11
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *  # type: ignore
from PySide2.QtGui import *  # type: ignore
from PySide2.QtWidgets import *  # type: ignore


class Ui_MainWindow(QMainWindow):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 600)
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QSize(800, 600))
        MainWindow.setMaximumSize(QSize(1280, 600))
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName(u"actionExit")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.Title = QLabel(self.centralwidget)
        self.Title.setObjectName(u"Title")
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.Title.setFont(font)
        self.Title.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.Title)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(-1, 6, -1, 6)
        self.Welcome_Message = QLabel(self.centralwidget)
        self.Welcome_Message.setObjectName(u"Welcome_Message")
        font1 = QFont()
        font1.setPointSize(13)
        self.Welcome_Message.setFont(font1)

        self.horizontalLayout.addWidget(self.Welcome_Message)

        self.RefreshButton = QPushButton(self.centralwidget)
        self.RefreshButton.setObjectName(u"RefreshButton")
        self.RefreshButton.setMaximumSize(QSize(200, 16777215))

        self.horizontalLayout.addWidget(self.RefreshButton)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line)

        self.CameraList = QListWidget(self.centralwidget)
        QListWidgetItem(self.CameraList)
        QListWidgetItem(self.CameraList)
        QListWidgetItem(self.CameraList)
        self.CameraList.setObjectName(u"CameraList")
        self.CameraList.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.CameraList.setResizeMode(QListView.Adjust)
        self.CameraList.setSpacing(2)
        self.CameraList.setSelectionRectVisible(False)

        self.verticalLayout.addWidget(self.CameraList)

        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.line_2)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(-1, 6, -1, 6)
        self.TrainButton = QPushButton(self.centralwidget)
        self.TrainButton.setObjectName(u"TrainButton")

        self.horizontalLayout_2.addWidget(self.TrainButton)

        self.TestButton = QPushButton(self.centralwidget)
        self.TestButton.setObjectName(u"TestButton")

        self.horizontalLayout_2.addWidget(self.TestButton)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 30))
        self.menubar.setDefaultUp(False)
        self.menubar.setNativeMenuBar(True)
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuFile.setToolTipsVisible(False)
        self.menuInfo = QMenu(self.menubar)
        self.menuInfo.setObjectName(u"menuInfo")
        self.menuInfo.setToolTipsVisible(False)
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuInfo.menuAction())
        self.menuFile.addAction(self.actionExit)

        self.retranslateUi(MainWindow)
        self.actionExit.triggered.connect(MainWindow.close)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Face Recognition System", None))
        self.actionExit.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
#if QT_CONFIG(shortcut)
        self.actionExit.setShortcut(QCoreApplication.translate("MainWindow", u"Alt+X", None))
#endif // QT_CONFIG(shortcut)
        self.Title.setText(QCoreApplication.translate("MainWindow", u"Face Recognition System v0.3", None))
        self.Welcome_Message.setText(QCoreApplication.translate("MainWindow", u"Select your camera:", None))
        self.RefreshButton.setText(QCoreApplication.translate("MainWindow", u"Refresh list", None))

        __sortingEnabled = self.CameraList.isSortingEnabled()
        self.CameraList.setSortingEnabled(False)
        ___qlistwidgetitem = self.CameraList.item(0)
        ___qlistwidgetitem.setText(QCoreApplication.translate("MainWindow", u"Test1", None));
        ___qlistwidgetitem1 = self.CameraList.item(1)
        ___qlistwidgetitem1.setText(QCoreApplication.translate("MainWindow", u"Test2", None));
        ___qlistwidgetitem2 = self.CameraList.item(2)
        ___qlistwidgetitem2.setText(QCoreApplication.translate("MainWindow", u"Test3", None));
        self.CameraList.setSortingEnabled(__sortingEnabled)

        self.TrainButton.setText(QCoreApplication.translate("MainWindow", u"Train network", None))
        self.TestButton.setText(QCoreApplication.translate("MainWindow", u"Test network", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuInfo.setTitle(QCoreApplication.translate("MainWindow", u"Info", None))
    # retranslateUi

