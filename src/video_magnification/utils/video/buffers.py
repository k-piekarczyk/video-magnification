import numpy as np
import numpy.typing as npt
import cv2

from typing import Optional, List, Tuple

from video_magnification.utils.video import VideoFileReader, prepare_for_laplace_pyramid, laplace_pyramid_step
from video_magnification.utils.math import get_max_pyramid_depth


__all__ = ["load_laplacian_pyramid_frames_to_buffers"]


def load_laplacian_pyramid_frames_to_buffers(
    vfr: VideoFileReader, depth: int, color_space: Optional[int] = None
) -> Tuple[List[npt.NDArray[np.uint8]], int]:
    height, width, max_frame_count = vfr.get_stats()
    max_depth = get_max_pyramid_depth(height=height, width=width)

    if depth > max_depth:
        raise Exception(f"Provided depth of {depth} is too high for the given source (max depth: {max_depth}.")

    pyramid_buffers: List[Optional[npt.NDArray[np.uint8]]] = [None for _ in range(max_depth + 1)]

    frame_count = 0
    cap = vfr.get_cap()
    while cap.isOpened():
        ret, raw_frame = cap.read()
        raw_frame: npt.NDArray[np.uint8]  # just here for type annotation

        if ret:
            frame = prepare_for_laplace_pyramid(frame=raw_frame)

            if color_space is not None:
                frame: npt.NDArray[np.uint8] = cv2.cvtColor(frame, code=color_space)

            scaled, diff = laplace_pyramid_step(frame=frame)
            diff_list = [diff]

            for i in range(1, depth):
                scaled, diff = laplace_pyramid_step(frame=scaled)
                diff_list.append(diff)

            for i in range(depth):
                if pyramid_buffers[i] is None:
                    x, y, p = diff_list[i].shape
                    pyramid_buffers[i] = np.ndarray((max_frame_count, x, y, p), dtype=np.uint8)
                pyramid_buffers[i][frame_count] = diff_list[i]

            if pyramid_buffers[depth] is None:
                x, y, p = scaled.shape
                pyramid_buffers[depth] = np.ndarray((max_frame_count, x, y, p), dtype=np.uint8)
            pyramid_buffers[depth][frame_count] = scaled

            frame_count = frame_count + 1
        else:
            break

    for i in range(depth + 1):
        pyramid_buffers[i] = pyramid_buffers[i][:frame_count]

    return pyramid_buffers, frame_count
