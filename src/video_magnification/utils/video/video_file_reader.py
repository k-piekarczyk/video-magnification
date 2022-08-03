import os
import cv2
from typing import Tuple

from .exceptions import VideoNotOpeningException, NotAVideoFileException, ClosedVideoFileReaderException


__all__ = ["VideoFileReader"]


class VideoFileReader:
    def __init__(self, filepath: str):
        abs_filepath = os.path.abspath(filepath)
        if not os.path.isfile(abs_filepath):
            raise NotAVideoFileException(filepath=abs_filepath)

        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        cap = cv2.VideoCapture(abs_filepath)

        # Check if camera opened successfully
        if cap.isOpened() is False:
            raise VideoNotOpeningException()

        self.cap = cap
        self.open = True

    def close(self):
        if not self.open:
            raise ClosedVideoFileReaderException()

        self.open = False
        self.cap.release()

    def get_cap(self):
        """
        Returns raw cv2.VideoCapture, closes the VideoFileReader
        """
        if not self.open:
            raise ClosedVideoFileReaderException()

        self.open = False
        return self.cap

    def get_stats(self) -> Tuple[int, int, int, float]:
        """
        Returns a tuple with video statistics: `(width, height, frame_count, fps)`
        """
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        return height, width, frame_count, fps
