from typing import Optional
import cv2
import sys
import numpy.typing as npt
import numpy as np

from video_magnification.utils.video import VideoFileReader, load_laplacian_pyramid_frames_to_buffers
from video_magnification.utils.math import get_max_pyramid_depth


def laplace():
    filepath = sys.argv[1]

    vfr = VideoFileReader(filepath=filepath)

    height, width, _ = vfr.get_stats()
    max_depth = get_max_pyramid_depth(height=height, width=width)
    DEPTH = max_depth

    pyramid_buffers, frame_count = load_laplacian_pyramid_frames_to_buffers(vfr=vfr, depth=DEPTH)

    stop = False
    while not stop:
        while not stop:
            for i in range(frame_count):
                for d in range(DEPTH + 1):
                    cv2.imshow(f"Depth: {d}", pyramid_buffers[d][i])

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop = True
                    break


def main():
    laplace()


if __name__ == "__main__":
    main()
