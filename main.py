from ultralytics import YOLO
from shapely.geometry import Point, LineString
import cv2
import mouse
import pyautogui
import numpy as np
import keyboard

def mn():
    screen_size = (1920, 1080)  # Replace with your screen resolution

    # Create a VideoWriter object to save the output as a video file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video compression
    out = cv2.VideoWriter('screen_capture.avi', fourcc, 20.0, screen_size)

    model = YOLO('models\\valo_1.pt')

    def yolobbox2bbox(li):
        x, y, w, h = li[0], li[1], li[2], li[3]
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2
        return [x1, y1, x2, y2]

    def aim(boxes):
        m_pos = mouse.get_position()
        dis = []
        if len(boxes) != 0:
            for i in boxes:
                coord1 = np.array([float(i[0]), float(i[1])])
                dis_ = np.linalg.norm(np.array(m_pos) - coord1)
                dis.append(abs(dis_))
            inde = dis.index(min(dis))

            sc_1 = [(float(boxes[inde][0]), float(boxes[inde][1])), (float(boxes[inde][0]), float(boxes[inde][3]))]
            sc_2 = [(float(boxes[inde][0]), float(boxes[inde][1])), (float(boxes[inde][2]), float(boxes[inde][1]))]

            line_1 = LineString(sc_1)
            line_2 = LineString(sc_2)
            divided_point_1 = line_1.interpolate(1 / 5, normalized=True)
            divided_point_2 = line_2.interpolate(1 / 2, normalized=True)
            print(mouse.get_position())
            print(divided_point_2.x,divided_point_1.y)
            mouse.move(float(divided_point_2.x),float(divided_point_1.y))

    while True:
        if keyboard.is_pressed('NUMLOCK') == True:
            img = pyautogui.screenshot()
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


            out.write(frame)
            result = model(frame)
            boxes_ = []
            for i in list(result[0].boxes.xywh):
                boxes_.append(yolobbox2bbox(i))

            aim(boxes_)

        if keyboard.is_pressed('ESC'):
            print('close_')
            break

        # Exit the loop if 'q' is pressed

    # Release the VideoWriter and close the OpenCV window
    out.release()
    cv2.destroyAllWindows()

mn()