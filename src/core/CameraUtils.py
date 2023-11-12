import cv2  # For camera processing

# Enumerate available cameras
# Link: https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python#62639343


def enumerateCamera():
    """
    Test the ports and returns a tuple with the available ports
    and the ones that are working.
    """
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print("Port %s is not working." % dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)"
                      % (dev_port, h, w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but \
                      does not reads." % (dev_port, h, w))
                available_ports.append(dev_port)
        dev_port += 1
    return available_ports, working_ports


# Simple function for watching camera feed

def openCameraFeed(camera_id: int):
    print("Opening camera: " + str(camera_id))
    camera = cv2.VideoCapture(camera_id)

    if camera.isOpened() is False:
        print("Failed to open camera.")
        return

    while True:
        ret, frame = camera.read()

        cv2.imshow("Video feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit requested.")
            break

    camera.release()
    cv2.destroyAllWindows()
