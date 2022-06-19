import cv2
import numpy as np
import numpy.typing as npt
from typing import Tuple


__all__ = ["scale_frame_down", "scale_frame_up", "laplace_pyramid_step"]


def scale_frame_down(frame, scaling_factor: int):
    """
    Scales the given frame down using a Gaussian pyramid
    """
    for _ in range(scaling_factor):
        frame = cv2.pyrDown(frame)

    return frame


def scale_frame_up(frame, scaling_factor: int):
    """
    Scales the given frame down using a Gaussian pyramid
    """
    for _ in range(scaling_factor):
        frame = cv2.pyrUp(frame)

    return frame

def laplace_pyramid_step(frame: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """
    Scales the given frame down and returns the scaleddown frame and it's diff from original
    """

    scaled_down = cv2.pyrDown(frame)
    expanded = cv2.pyrUp(scaled_down)

    diff = cv2.subtract(frame, expanded)

    return (scaled_down, diff)
    



