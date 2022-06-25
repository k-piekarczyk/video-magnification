import cv2
import sys

from video_magnification.utils.video import (
    VideoFileReader,
    load_laplacian_pyramid_frames_to_buffers,
    merge_laplacian_pyramid_frames_into_single_buffer,
)
from video_magnification.utils.math import get_max_pyramid_depth


def laplace():
    filepath = sys.argv[1]

    vfr = VideoFileReader(filepath=filepath)

    height, width, _ = vfr.get_stats()
    max_depth = get_max_pyramid_depth(height=height, width=width)
    DEPTH = max_depth

    pyramid_buffers, frame_count = load_laplacian_pyramid_frames_to_buffers(vfr=vfr, depth=DEPTH)

    buffer = merge_laplacian_pyramid_frames_into_single_buffer(pyramid_buffers=pyramid_buffers)

    stop = False
    while not stop:
        while not stop:
            for i in range(frame_count):

                cv2.imshow("Merged", buffer[i])

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop = True
                    break


def main():
    laplace()


if __name__ == "__main__":
    main()
