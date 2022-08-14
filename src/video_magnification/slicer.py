from turtle import shape
import cv2
import sys

import numpy as np

from typing import List, Tuple, Optional

from video_magnification.utils.video import VideoFileReader

section: List[Tuple[int, int]] = []
mouse_position: Optional[Tuple[int, int]] = None

def mouse_events(event, x, y, flags, param):
            global section, mouse_position
            if len(section) > -1 and len(section) < 2:
                if event == cv2.EVENT_LBUTTONDOWN:
                    section.append((x, y))
                elif event == cv2.EVENT_LBUTTONUP:
                    section.append((section[0][0], y))
            if event == cv2.EVENT_MOUSEMOVE:
                mouse_position = (x, y)

def slicer(filepath: str, output: str, selected: Optional[List[Tuple[int, int]]] = None):
    global section, mouse_position

    vfr = VideoFileReader(filepath=filepath)
    _, _, frame_count, _ = vfr.get_stats()
    cap = vfr.get_cap()

    ret, frame = cap.read()
    if selected is not None:
        section = selected
    elif ret:
        cv2.namedWindow("First frame")
        cv2.setMouseCallback("First frame", mouse_events)

        while True:
            copy = frame.copy()
            if (len(section) == 1):
                copy = cv2.circle(copy, section[0], radius=1, color=(255, 0, 255), thickness=5)
                copy = cv2.line(copy, section[0], (section[0][0], mouse_position[1]), color=(255, 0, 255), thickness=2)
            if (len(section) == 2):
                copy = cv2.circle(copy, section[0], radius=1, color=(0, 255, 0), thickness=5)
                copy = cv2.circle(copy, section[1], radius=1, color=(0, 255, 0), thickness=5)
                copy = cv2.line(copy, section[0], section[1], color=(0, 255, 0), thickness=2)

            cv2.imshow("First frame", copy)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("You quit the program")
                exit(0)
            elif key == ord("d"):
                if len(section) < 2:
                    print("You need to select a section before proceeding")
                    continue
                print(f"Section selected: start {section[0]}; end {section[1]}")
                cv2.destroyWindow("First frame")
                break
            elif key == ord("r"):
                section = []
                mouse_position = None
    else:
        raise Exception("Something went wrong with reading the video")
    
    if section[0][1] > section[1][1]:
        np.roll(section, 1)

    height = section[1][1] - section[0][1]
    
    sliced = np.ndarray(shape=(height, frame_count, frame.shape[2]))

    sliced[:, 0] = frame[section[0][1]:section[1][1], section[0][0]]

    counter = 1
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            sliced[:, counter, :] = frame[section[0][1]:section[1][1], section[0][0], :]
            counter += 1
        else:
            break
    
    cv2.imwrite(output, sliced)

def run():
    filepath = sys.argv[1]
    output = sys.argv[2]
    # slicer(filepath=filepath, output=output)
    slicer(filepath=filepath, output=output, selected=[(303, 48), (303, 148)])

if __name__ == "__main__":
    filepath = sys.argv[1]
    output = sys.argv[2]
    slicer(filepath=filepath, output=output)