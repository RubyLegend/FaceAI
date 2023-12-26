import cv2  # For camera processing
import os
import time
import math
import numpy as np
import threading
from copy import deepcopy

from PySide2 import QtCore, QtGui
from PySide2.QtCore import Qt, QSize
from PySide2.QtWidgets import QListWidgetItem, QListWidget
from PySide2.QtGui import QImage, QPixmap

from core.headPosition import detectFace
from core.network import getNewModel, getNewModel2, loadModel, train, modelDetectFace, get_offsets

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

    print("[DEBUG] Opening camera: " + str(camera_path))
    camera = cv2.VideoCapture(camera_path)

    if camera.isOpened() is False:
        errorSignal.errorSignal.emit("Failed to open camera. Error occured.")
        return

    # Forcing maximum camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    window_width, window_height = 1280, 720

    VideoPreviewWindow.show()
    VideoPreviewWindow.resize(window_width, window_height)
    VideoPreviewWindow.setMaximumSize(QSize(window_width, window_height))
    VideoPreviewWindow.setMinimumSize(QSize(window_width, window_height))

    reading = True

    while reading:
        ret, frame = camera.read()

        # frame = detectFace(frame)

        # Swapping to match RGB888, othervise coming as BGR888
        # Which is not working on pyside2 from a pypi repo
        # frame[..., [0, 2]] = frame[..., [2, 0]]
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        
        VideoPreviewWindow.ImageFeed.setPixmap(QPixmap(QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format.Format_RGB888)).scaled(QSize(window_width, window_height), Qt.KeepAspectRatio))
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
        # item.setTextAlignment(Qt.AlignVCenter)


def enrollNewFace(camera: QListWidgetItem, VideoPreviewWindow, FacePreviewWindow, app):
    camera_name = camera.text()
    camera_path = camera_name.split(':')[0]

    print("[DEBUG] Opening camera: " + str(camera_path))
    camera = cv2.VideoCapture(camera_path)

    if camera.isOpened() is False:
        errorSignal.errorSignal.emit("Failed to open camera. Error occured.")
        return

    img_w = 1280
    img_h = 720
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, img_w)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, img_h)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    window_width, window_height = 1280, 720
    facewindow_width, facewindow_height = 500, 500

    VideoPreviewWindow.show()
    VideoPreviewWindow.resize(window_width, window_height)
    VideoPreviewWindow.setMinimumSize(QSize(0, 0))
    VideoPreviewWindow.setMaximumSize(QSize(window_width, window_height))
    VideoPreviewWindow.setMinimumSize(QSize(window_width, window_height))

    FacePreviewWindow.show()
    FacePreviewWindow.resize(facewindow_width, facewindow_height)
    FacePreviewWindow.setMinimumSize(QSize(0, 0))
    FacePreviewWindow.setMaximumSize(QSize(facewindow_width, facewindow_height))
    FacePreviewWindow.setMinimumSize(QSize(facewindow_width, facewindow_height))

    circle_center = (int(img_w/2), int(img_h/2))
    radius = 250

    fill_color = (255, 255, 255)
    fill_color_not_registered = (200, 200, 200)
    fill_color_registered = (0, 255, 0)
    mask_value = 255
    stencil = np.zeros((img_h, img_w), dtype=np.int16)
    cv2.circle(stencil, circle_center, radius, (255, 255, 255), -1)
    sel = stencil != mask_value  # select everything that is not mask_value

    # Prepare lines that will indicate, whether specified angle
    # was registered
    angles = [(350 - alpha, (int(circle_center[0] + radius * math.cos(alpha*2*math.pi / 360)),
                int(circle_center[1] + radius * math.sin(alpha*2*math.pi / 360))), False) for alpha in range(0, 360, 10)]
    angles2 = [(int(val[0] + math.cos((350 - alpha)*2*math.pi/360)*20), int(val[1] + math.sin((350 - alpha)*2*math.pi/360)*20)) for alpha, val, _ in angles]

    dots_selection = []
    face = None

    for i in range(len(angles)):
        stencil = np.zeros((img_h, img_w), dtype=np.int16)
        cv2.line(stencil, angles[i][1], angles2[i], (255, 255, 255), 3)
        sel2 = stencil == mask_value
        dots_selection.append(sel2)

    reading = False  # Change to False, if you want to temporary skip capturing phase

    while reading:
        ret, raw_frame = camera.read()
        if face is None:
            face = raw_frame

        frame, angle, detected, x, y = detectFace(raw_frame, draw=True)
        # Trying to recognize again, because later network fails to recognize it
        # Even though it was recognized successfully.
        _, _, detected2, x_coords, y_coords = detectFace(cv2.resize(raw_frame, (640, 360)), draw=False)
        it = len(angles) - int(angle / 10) - 1
        if angle != 0 and int(angle % 10) <= 3 and angles[it][2] is not True and detected2 is True:
            print(f'Detected on downscaled image: {detected2}')
            print(f'Registering: {it}')
            original = angles[it]
            angles[it] = (original[0], original[1], True)
            np.save(f"./src/data/{it}", raw_frame)
            
            x1, x2, y1, y2, is_resize = get_offsets(x_coords[0], x_coords[1], y_coords[0], y_coords[1])
            with open(f"./src/data/{it}.pos", "w") as file:
                file.write(str(x1) + '\n')
                file.write(str(x2) + '\n')
                file.write(str(y1) + '\n')
                file.write(str(y2) + '\n')
                file.write(str(is_resize) + '\n')

        # Swapping to match RGB888, othervise coming as BGR888
        # Which is not working on pyside2 from a pypi repo
        # frame[..., [0, 2]] = frame[..., [2, 0]]

        # Getting face cropped
        if detected:
            # face = deepcopy(frame)
            face = raw_frame[y[0]:y[1], x[0]:x[1]]
            if face.shape[0] != 0 and face.shape[1] != 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (400, 400))
            # face = np.ascontiguousarray(face)
            print(face.shape)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Drawing white frame, to limit position of face in frame
        frame[sel] = fill_color
        # Drawing gray dots, to mimic FaceID dots
        reading = False
        for anglet, _, _ in angles:
            it = len(angles) - int(anglet / 10) - 1
            if angles[it][2] is True:
                frame[dots_selection[it]] = fill_color_registered
            else:
                frame[dots_selection[it]] = fill_color_not_registered
                reading = True

        FacePreviewWindow.show()
        VideoPreviewWindow.ImageFeed.setPixmap(QPixmap(QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format.Format_RGB888)).scaled(QSize(window_width, window_height), Qt.KeepAspectRatio))
        FacePreviewWindow.ImageFeed.setPixmap(QPixmap(QImage(face.data, face.shape[1], face.shape[0], QtGui.QImage.Format.Format_RGB888)).scaled(QSize(facewindow_width, facewindow_height), Qt.KeepAspectRatio))
        app.processEvents()

        if not VideoPreviewWindow.isVisible():
            reading = False

    camera.release()
    FacePreviewWindow.hide()

    if VideoPreviewWindow.isVisible():
        event = threading.Event()
        threading.Thread(target=train, args=(getNewModel2(), event)).start()
        learning = True
        counter = 0

        while learning:
            VideoPreviewWindow.ImageFeed.setText(f"Learning{'.' * int(counter % 4)}")
            app.processEvents()
            time.sleep(0.5)
            counter += 1
            if counter == 4:
                counter = 0

            if event.is_set():
                learning = False

        VideoPreviewWindow.ImageFeed.setText("Done. Now you can test by selecting \'Test network\' button")
        app.processEvents()
        app.processEvents()
        time.sleep(3)

    VideoPreviewWindow.hide()
    FacePreviewWindow.hide()


def testFaceRealTime(camera: QListWidgetItem, VideoPreviewWindow, FacePreviewWindow, app):
    camera_name = camera.text()
    camera_path = camera_name.split(':')[0]

    print("[DEBUG] Opening camera: " + str(camera_path))
    camera = cv2.VideoCapture(camera_path)

    if camera.isOpened() is False:
        errorSignal.errorSignal.emit("Failed to open camera. Error occured.")
        return

    # Forcing maximum camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    window_width, window_height = 1280, 720
    facewindow_width, facewindow_height = 600, 600

    VideoPreviewWindow.show()
    VideoPreviewWindow.resize(window_width, window_height)
    VideoPreviewWindow.setMinimumSize(QSize(0, 0))
    VideoPreviewWindow.setMaximumSize(QSize(window_width, window_height))
    VideoPreviewWindow.setMinimumSize(QSize(window_width, window_height))

    FacePreviewWindow.show()
    FacePreviewWindow.resize(facewindow_width, facewindow_height)
    FacePreviewWindow.setMinimumSize(QSize(0, 0))
    FacePreviewWindow.setMaximumSize(QSize(facewindow_width, facewindow_height))
    FacePreviewWindow.setMinimumSize(QSize(facewindow_width, facewindow_height))

    face = None

    # Loading model
    print("[DEBUG] Loading model...")
    model = loadModel()
    print("[DEBUG] Model loaded.")

    reading = True  # Change to False, if you want to temporary skip capturing phase

    while reading:
        ret, frame = camera.read()
        if face is None:
            face = frame

        _, _, detected, x_coords, y_coords = detectFace(frame, draw=False)
        result, confidence = modelDetectFace(model, frame)
        print(result)

        # Swapping to match RGB888, othervise coming as BGR888
        # Which is not working on pyside2 from a pypi repo
        # frame[..., [0, 2]] = frame[..., [2, 0]]

        if detected:
            # face = deepcopy(frame)
            face = frame[y_coords[0]:y_coords[1], x_coords[0]:x_coords[1]]
            if face.shape[0] != 0 and face.shape[1] != 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (400, 400))
            # face = np.ascontiguousarray(face)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if detected:
            if result is True:
                cv2.rectangle(frame, (x_coords[0], y_coords[0]), (x_coords[1], y_coords[1]), (0, 255, 0), 1)
                cv2.putText(frame, f"Conf: {confidence[0][0]:0.3f}", (x_coords[0], y_coords[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x_coords[0], y_coords[0]), (x_coords[1], y_coords[1]), (255, 0, 0), 1)
                cv2.putText(frame, f"Conf: {confidence[0][0]:0.3f}", (x_coords[0], y_coords[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        FacePreviewWindow.show()
        VideoPreviewWindow.ImageFeed.setPixmap(QPixmap(QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format.Format_RGB888)))
        FacePreviewWindow.ImageFeed.setPixmap(QPixmap(QImage(face.data, face.shape[1], face.shape[0], QtGui.QImage.Format.Format_RGB888)).scaled(QSize(facewindow_width, facewindow_height), Qt.KeepAspectRatio))
        app.processEvents()

        if not VideoPreviewWindow.isVisible():
            reading = False

    camera.release()
    VideoPreviewWindow.hide()
    FacePreviewWindow.hide()
