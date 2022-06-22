import cv2
import numpy as np
import numpy.typing as npt
from typing import Tuple

from ..math.powers import closest_power_of_2, is_power_of_2


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


def prepare_for_laplace_pyramid(
    frame: npt.NDArray[np.uint8], min_unit_size: int = 8
) -> Tuple[npt.NDArray[np.uint8], int]:
    """
    Prepares a frame in each side has a length of a power of two, and returns the maximum depth for that image,
    for a given `min_unit_size`, which denotes how big is the lowest level of the pyramid (by default, 8x8)
    """
    if min_unit_size <= 2 or not is_power_of_2(min_unit_size):
        raise Exception(f"'min_unit_size' has to be at least 2, and be divisible by 2. Provided value: {min_unit_size}")

    height, width, channels = frame.shape

    adjusted_height = closest_power_of_2(height)
    adjusted_width = closest_power_of_2(width)

    smaller_adjusted_side_length = min(adjusted_height, adjusted_width)
    if smaller_adjusted_side_length <= min_unit_size:
        raise Exception(
            f"Impossible to build a Laplacian pyramid with the given 'min_unit_size' of {min_unit_size} from the given 'frame'"
        )

    max_depth = smaller_adjusted_side_length.bit_length() - min_unit_size.bit_length()

    adjusted_shape = (adjusted_height, adjusted_width, channels)

    adjusted_frame: npt.NDArray[np.uint8] = np.ndarray(shape=adjusted_shape, dtype=np.uint8)

    adjusted_frame[:height, :width] = frame

    adjusted_frame[height:, :width] = frame[height - 1 :, :]
    adjusted_frame[:height, width:] = frame[:, width - 1 :]
    adjusted_frame[height:, width:] = frame[height - 1 :, width - 1 :]

    return adjusted_frame, max_depth


def laplace_pyramid_step(frame: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """
    Scales the given frame down and returns the scaleddown frame and it's diff from original
    """
    scaled_down = cv2.pyrDown(frame)
    expanded = cv2.pyrUp(scaled_down)

    diff = cv2.subtract(frame, expanded)

    return (scaled_down, diff)
