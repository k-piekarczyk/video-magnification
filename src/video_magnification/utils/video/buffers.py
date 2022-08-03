import numpy as np
import numpy.typing as npt
import cv2

from typing import Optional, List, Tuple, Any

from video_magnification.utils.video import VideoFileReader, prepare_for_laplace_pyramid, laplace_pyramid_step
from video_magnification.utils.math import get_max_pyramid_depth


__all__ = [
    "load_laplacian_pyramid_frames_to_buffers",
    "merge_laplacian_pyramid_frames_into_single_buffer",
    "load_frames_to_gaussian_buffer",
]


def load_laplacian_pyramid_frames_to_buffers(
    vfr: VideoFileReader, depth: int, color_space: Optional[int] = None, single_channel: Optional[int] = None
) -> Tuple[List[npt.NDArray[np.uint8]], int]:
    height, width, max_frame_count, _ = vfr.get_stats()
    max_depth = get_max_pyramid_depth(height=height, width=width)

    if depth > max_depth:
        raise Exception(f"Provided depth of {depth} is too high for the given source (max depth: {max_depth})")

    if single_channel is not None and (single_channel < 0 or single_channel > 2):
        raise Exception(f"Provided single channel index of {single_channel} is invalid (valid indexes: 0, 1 or 2)")

    pyramid_buffers: List[Optional[npt.NDArray[np.uint8]]] = [None for _ in range(depth + 1)]

    frame_count = 0
    cap = vfr.get_cap()
    while cap.isOpened():
        cap_read: Tuple[Any, npt.NDArray[np.uint8]] = cap.read()
        ret = cap_read[0]
        raw_frame = cap_read[1]

        if ret:
            frame = prepare_for_laplace_pyramid(frame=raw_frame)

            if color_space is not None:
                frame = cv2.cvtColor(frame, code=color_space)

            if single_channel is not None:
                frame = frame[:, :, single_channel]

            scaled, diff = laplace_pyramid_step(frame=frame)
            diff_list = [diff]

            for i in range(1, depth):
                scaled, diff = laplace_pyramid_step(frame=scaled)
                diff_list.append(diff)

            for i in range(depth):
                if pyramid_buffers[i] is None:
                    if single_channel is None:
                        x, y, p = diff_list[i].shape
                        pyramid_buffers[i] = np.ndarray((max_frame_count, x, y, p), dtype=np.uint8)
                    else:
                        x, y = diff_list[i].shape
                        pyramid_buffers[i] = np.ndarray((max_frame_count, x, y), dtype=np.uint8)
                pyramid_buffers[i][frame_count] = diff_list[i]

            if pyramid_buffers[depth] is None:
                if single_channel is None:
                    x, y, p = scaled.shape
                    pyramid_buffers[depth] = np.ndarray((max_frame_count, x, y, p), dtype=np.uint8)
                else:
                    x, y = scaled.shape
                    pyramid_buffers[depth] = np.ndarray((max_frame_count, x, y), dtype=np.uint8)
            pyramid_buffers[depth][frame_count] = scaled

            frame_count = frame_count + 1
        else:
            break

    for i in range(depth + 1):
        pyramid_buffers[i] = pyramid_buffers[i][:frame_count]

    return pyramid_buffers, frame_count


def merge_laplacian_pyramid_frames_into_single_buffer(
    pyramid_buffers: List[npt.NDArray[np.uint8]], color_space: Optional[int] = None, ignore_last_level: bool = False
) -> npt.NDArray[np.uint8]:

    if ignore_last_level:
        pyramid_buffers = pyramid_buffers[:-1]

    depth = len(pyramid_buffers)

    shape = pyramid_buffers[0].shape

    if len(shape) == 3:
        frame_count, height, width = shape
        buffer = np.ndarray((frame_count, height, width), dtype=np.uint8)
    else:
        frame_count, height, width, channels = shape
        buffer = np.ndarray((frame_count, height, width, channels), dtype=np.uint8)

    for i in range(frame_count):
        frame = pyramid_buffers[-1][i]

        for j in reversed(range(depth - 1)):
            frame = cv2.pyrUp(frame)
            diff = pyramid_buffers[j][i]

            frame = diff + frame

        if color_space is None:
            buffer[i] = frame
        else:
            buffer[i] = cv2.cvtColor(frame, code=color_space)

        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    return buffer


def load_frames_to_gaussian_buffer(
    vfr: VideoFileReader, depth: int, color_space: Optional[int] = None
) -> Tuple[npt.NDArray[np.uint8], int]:
    _, _, max_frame_count, _ = vfr.get_stats()

    buffer: Optional[npt.NDArray[np.uint8]] = None

    frame_count = 0
    cap = vfr.get_cap()
    while cap.isOpened():
        ret, raw_frame = cap.read()
        raw_frame: npt.NDArray[np.uint8]  # just here for type annotation

        if ret:
            frame = raw_frame

            if color_space is not None:
                frame: npt.NDArray[np.uint8] = cv2.cvtColor(raw_frame, code=color_space)

            for _ in range(depth):
                frame = cv2.pyrDown(frame)

            if buffer is None:
                x, y, p = frame.shape
                buffer = np.ndarray((max_frame_count, x, y, p), dtype=np.uint8)
            buffer[frame_count] = frame

            frame_count = frame_count + 1
        else:
            break

    buffer = buffer[:frame_count]

    return buffer, frame_count
