from typing import Any, Tuple
import cv2
import sys
import numpy.typing as npt
import numpy as np

from video_magnification.processing import process_frame_buffer
from video_magnification.utils.video import VideoFileReader, scale_frame_up, laplace_pyramid_step

SCALE_FACTOR = 3
MAGNIFICATION_FACTOR = 4


def processed_run():
    filepath = sys.argv[1]

    vf_reader = VideoFileReader(filepath=filepath)

    frame_buffer = vf_reader.load_frames_into_buffer(SCALE_FACTOR)

    processed_frame_buffer = process_frame_buffer(frame_buffer=frame_buffer, magnification_factor=MAGNIFICATION_FACTOR)

    cap = VideoFileReader(filepath=filepath).get_cap()
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            scaled_up = scale_frame_up(frame=processed_frame_buffer[frame_count], scaling_factor=SCALE_FACTOR)[:-6,:,:].astype("uint8")

            cv2.imshow("Original", frame)
            cv2.imshow("Redone", scaled_up)

            alpha = .95
            beta = 1 - alpha

            cv2.imshow("Joined", cv2.addWeighted(frame, alpha, scaled_up, beta, 0.0))

            frame_count = frame_count + 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

def laplace():
    filepath = sys.argv[1]

    vf_reader = VideoFileReader(filepath=filepath)

    cap = VideoFileReader(filepath=filepath).get_cap()

    _, frame = cap.read()
    frame: npt.NDArray[np.uint8]  # just here for type annotation
    
    scaled, diff = laplace_pyramid_step(frame=frame)

    cv2.imshow("Original", frame)
    cv2.imshow("Scaled", scaled)
    cv2.imshow("Diff", diff)


    cv2.waitKey(0)
    



def main():
    # processed_run()
    laplace()


if __name__ == "__main__":
    main()
