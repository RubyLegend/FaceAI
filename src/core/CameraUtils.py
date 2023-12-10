import cv2  # For camera processing
import os
import time

from PySide2 import QtCore, QtGui
from PySide2.QtCore import Qt, QSize
from PySide2.QtWidgets import QListWidgetItem, QListWidget
from PySide2.QtGui import QImage, QPixmap

# Enumerate available cameras
# Custom Linux and Windows detection, due to missing generic way


def enumerateCameras_Linux():
    cameras = {}
    i = 0  
    video_id = 0 # Iterate over each /dev/video*, but under /sys/class/video4linux
    # Assume, that video4linux is installed, so that path will exist.
    device_exist = True
    while device_exist:
        if os.path.isfile(f"/sys/class/video4linux/video{i}/name"):
            # print(f"Found path: /sys/class/video4linux/video{i}")
            with open(f"/sys/class/video4linux/video{i}/name") as file:
                name = file.read().strip()
                # print(name)
                cameras[i] = name
                i += 1
        else:
            device_exist = False

        video_id += 1

    return cameras


def enumerateCameras():
    if os.name == "posix":  # *nix based machine
        return enumerateCameras_Linux()
    else:
        print("Windows support is under development")
        return {}


# Signal to open error window
class errorObject(QtCore.QObject):
    errorSignal = QtCore.Signal(str)


errorSignal = errorObject()


# Simple function for watching camera feed
def openCameraFeed(camera: QListWidgetItem, VideoPreviewWindow, app):
    camera_name = camera.text()
    camera_path = camera_name.split(':')[0]

    print("Opening camera: " + str(camera_path))
    camera = cv2.VideoCapture(camera_path)

    if camera.isOpened() is False:
        errorSignal.errorSignal.emit("Failed to open camera. Error occured.")
        return

    # Forcing maximum camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    VideoPreviewWindow.show()
    VideoPreviewWindow.resize(1280,720)
    VideoPreviewWindow.setMinimumSize(QSize(1280, 720))
    VideoPreviewWindow.setMaximumSize(QSize(1280, 720))
    
    reading = True
    
    while reading:
        ret, frame = camera.read()

        VideoPreviewWindow.ImageFeed.setPixmap(QPixmap(QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)))
        app.processEvents()

        if not VideoPreviewWindow.isVisible():
            reading = False

    camera.release()
    VideoPreviewWindow.hide()


def fillCameraList(CameraList: QListWidget):
    CameraList.clear()
    available_cameras = enumerateCameras()
    for port in available_cameras:
        QListWidgetItem(CameraList)
        item = CameraList.item(port)
        item.setText(f'/dev/video{port}: {available_cameras[port]}')
        item.setTextAlignment(Qt.AlignVCenter)
