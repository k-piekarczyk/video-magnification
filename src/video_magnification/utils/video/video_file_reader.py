import os
import cv2
import numpy as np

from .exceptions import VideoNotOpeningException, NotAVideoFileException, ClosedVideoFileReaderException
from .manipulation import scale_frame_down


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

    def load_frames_into_buffer(self, scaling_factor: int = 0):
        """
        Loads the video into a frame buffer (scaled appropriately) and returns the buffer
        """
        if not self.open:
            raise ClosedVideoFileReaderException()

        frame_buffer = None
        while self.cap.isOpened():
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if ret is True:
                scaled_frame = scale_frame_down(frame=frame, scaling_factor=scaling_factor)

                if frame_buffer is None:
                    x, y, p = scaled_frame.shape
                    frame_buffer = np.ndarray((0, x, y, p))

                frame_buffer = np.append(frame_buffer, scaled_frame[np.newaxis, ...], 0)

            # Break the loop
            else:
                break

        self.close()

        return frame_buffer
