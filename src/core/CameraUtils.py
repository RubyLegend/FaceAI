import cv2  # For camera processing
import os

from PySide2 import QtCore
from PySide2.QtWidgets import QListWidgetItem

# Enumerate available cameras
# Custom Linux and Windows detection, due to missing generic way

def enumerateCameras_Linux():
    cameras = {}
    i = 0  # Iterate over each /dev/video*, but under /sys/class/video4linux
    # Assume, that video4linux is installed
    device_exist = True
    while device_exist:
        if os.path.isfile(f"/sys/class/video4linux/video{i}/name"):
            print(f"Found path: /sys/class/video4linux/video{i}")
            with open(f"/sys/class/video4linux/video{i}/name") as file:
                name = file.read()
                print(name)
                cameras[i] = name
        else:
            device_exist = False

        i = i + 1

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


def openCameraFeed(camera: QListWidgetItem):
    camera_name = camera.text()
    camera_path = camera_name.split(':')[0]

    print("Opening camera: " + str(camera_path))
    camera = cv2.VideoCapture(camera_path)

    if camera.isOpened() is False:
        errorSignal.errorSignal.emit("Failed to open camera. Error occured.")
        return
    
    for i in range(0, 10):
        print(f"{i}: {camera.get(i)}")


    # Forcing maximum camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    while True:
        ret, frame = camera.read()

        cv2.imshow("Video feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit requested.")
            break

    camera.release()
    cv2.destroyAllWindows()
