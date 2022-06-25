import cv2
import numpy as np
import numpy.typing as npt
from typing import Tuple

from ..math.powers import closest_power_of_2


__all__ = ["scale_frame_down", "scale_frame_up", "laplace_pyramid_step", "prepare_for_laplace_pyramid"]


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


def prepare_for_laplace_pyramid(frame: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """
    Prepares a frame in each side has a length of a power of two, and returns the maximum depth for that image,
    for a given `min_unit_size`, which denotes how big is the lowest level of the pyramid (by default, 8x8)
    """
    height, width, channels = frame.shape

    adjusted_height = closest_power_of_2(height)
    adjusted_width = closest_power_of_2(width)

    adjusted_shape = (adjusted_height, adjusted_width, channels)

    adjusted_frame: npt.NDArray[np.uint8] = np.ndarray(shape=adjusted_shape, dtype=np.uint8)

    adjusted_frame[:height, :width] = frame

    adjusted_frame[height:, :width] = frame[height - 1 :, :]
    adjusted_frame[:height, width:] = frame[:, width - 1 :]
    adjusted_frame[height:, width:] = frame[height - 1 :, width - 1 :]

    return adjusted_frame


def laplace_pyramid_step(frame: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """
    Scales the given frame down and returns the scaleddown frame and it's diff from original
    """
    scaled_down = cv2.pyrDown(frame)
    expanded = cv2.pyrUp(scaled_down)

    diff = cv2.subtract(frame, expanded)

    return (scaled_down, diff)
