import cv2
import sys

from video_magnification.processing import process_frame_buffer
from video_magnification.utils.video import VideoFileReader, scale_frame_up

SCALE_FACTOR = 3
MAGNIFICATION_FACTOR = 1


def main():
    filepath = sys.argv[1]

    vf_reader = VideoFileReader(filepath=filepath)

    frame_buffer = vf_reader.load_frames_into_buffer(SCALE_FACTOR)

    processed_frame_buffer = process_frame_buffer(frame_buffer=frame_buffer, magnification_factor=MAGNIFICATION_FACTOR)

    cap = VideoFileReader(filepath=filepath).get_cap()
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            scaled_up = scale_frame_up(frame=processed_frame_buffer[frame_count], scaling_factor=SCALE_FACTOR)

            cv2.imshow("Original", frame)
            cv2.imshow("Redone", scaled_up)

            frame_count = frame_count + 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break


if __name__ == "__main__":
    main()
