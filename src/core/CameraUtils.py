import cv2  # For camera processing
import os
import time
import math
import numpy as np
import threading

from PySide2 import QtCore, QtGui
from PySide2.QtCore import Qt, QSize
from PySide2.QtWidgets import QListWidgetItem, QListWidget
from PySide2.QtGui import QImage, QPixmap

from core.headPosition import detectFace
from core.network import getNewModel, loadModel, train, modelDetectFace

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


def get_offsets(x1, x2, y1, y2):
    left, right, up, down = 0,0,0,0

    if x2 - x1 != 200:
        left = int((200 - x2 + x1) / 2)
        # print(f"left: {left}")
        x1 -= left
        # print(f"new x1: {x1}")
        if x1 < 0:
            x1 = 0
            # print('x1 corrected to 0')
        x2 = x1 + 200
        if x2 > 640:  # TODO: add normal limits instead of hardcoded
            # print(f'x2: {x2} > 640, correcting')
            x1 -= (x2 - 640)
            x2 = 640
            # print(f'new x1: {x1}, x2 = {x2}')

    if y2 - y1 != 200:
        down = int((200 - y2 + y1) / 2)
        # print(f'down: {down}')
        y1 -= down
        # print(f'new y1: {y1}')
        if y1 < 0:
            y1 = 0
            # print('y1 corrected to 0')
        y2 = y1 + 200
        if y2 > 360:
            # print(f'y2: {y2} > 360, correcting')
            y1 -= (y2 - 360)
            y2 = 360
            # print(f'new y1: {y1}, y2 = {y2}')


    return x1, x2, y1, y2


def enrollNewFace(camera: QListWidgetItem, VideoPreviewWindow, app):
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

    VideoPreviewWindow.show()
    VideoPreviewWindow.resize(window_width, window_height)
    VideoPreviewWindow.setMaximumSize(QSize(window_width, window_height))
    VideoPreviewWindow.setMinimumSize(QSize(window_width, window_height))

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

    for i in range(len(angles)):
        stencil = np.zeros((img_h, img_w), dtype=np.int16)
        cv2.line(stencil, angles[i][1], angles2[i], (255, 255, 255), 3)
        sel2 = stencil == mask_value
        dots_selection.append(sel2)

    reading = True  # Change to False, if you want to temporary skip capturing phase

    while reading:
        ret, raw_frame = camera.read()

        frame, angle, detected, _, _ = detectFace(raw_frame, draw=True)
        # Trying to recognize again, because later network fails to recognize it
        # Even though it was recognized successfully.
        _, _, detected2, x_coords, y_coords = detectFace(cv2.resize(raw_frame, (640, 360)), draw=False)
        it = len(angles) - int(angle / 10) - 1
        if angle != 0 and int(angle % 10) <= 3 and angles[it][2] is not True and detected2 is True:
            print(f'Downscaled image: {detected2}')
            print(f'Registering {it}')
            original = angles[it]
            angles[it] = (original[0], original[1], True)
            np.save(f"./data/{it}", raw_frame)
            
            x1, x2, y1, y2 = get_offsets(x_coords[0], x_coords[1], y_coords[0], y_coords[1])
            with open(f"./data/{it}.pos", "w") as file:
                file.write(str(x1) + '\n')
                file.write(str(x2) + '\n')
                file.write(str(y1) + '\n')
                file.write(str(y2) + '\n')

        # Swapping to match RGB888, othervise coming as BGR888
        # Which is not working on pyside2 from a pypi repo
        # frame[..., [0, 2]] = frame[..., [2, 0]]
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

        VideoPreviewWindow.ImageFeed.setPixmap(QPixmap(QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format.Format_RGB888)).scaled(QSize(window_width, window_height), Qt.KeepAspectRatio))
        app.processEvents()

        if not VideoPreviewWindow.isVisible():
            reading = False

    camera.release()

    if VideoPreviewWindow.isVisible():
        event = threading.Event()
        threading.Thread(target=train, args=(getNewModel(), event)).start()
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


def testFaceRealTime(camera: QListWidgetItem, VideoPreviewWindow, app):
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

    # Loading model
    print("[DEBUG] Loading model...")
    model = loadModel()
    print("[DEBUG] Model loaded.")

    reading = True  # Change to False, if you want to temporary skip capturing phase

    while reading:
        ret, frame = camera.read()

        _, _, _, x_coords, y_coords = detectFace(frame, draw=False)

        frame2 = frame

        result = modelDetectFace(model, frame2)
        print(result)

        # Swapping to match RGB888, othervise coming as BGR888
        # Which is not working on pyside2 from a pypi repo
        # frame[..., [0, 2]] = frame[..., [2, 0]]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.rectangle(frame, (window_width - x_coords[0], y_coords[0]), (window_width - x_coords[1], y_coords[1]), 1)

        VideoPreviewWindow.ImageFeed.setPixmap(QPixmap(QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format.Format_RGB888)))
        app.processEvents()

        if not VideoPreviewWindow.isVisible():
            reading = False

    camera.release()
    VideoPreviewWindow.hide()
