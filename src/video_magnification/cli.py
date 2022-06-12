import cv2
import numpy as np
import sys
import os.path

from video_magnification.processing import process_chunk

SCALE_FACTOR = 3
MAGNIFICATION_FACTOR = 50


def main():
    filepath = sys.argv[1]
    abs_filepath = os.path.abspath(filepath)
    if not os.path.isfile(abs_filepath):
        raise Exception(f"Provided filepath {abs_filepath} is not a video file")

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(abs_filepath)

    # Check if camera opened successfully
    if cap.isOpened() is False:
        print("Error opening video stream or file")

    # Read until video is completed
    frame_buffer = None
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret is True:
            scaled_down = frame

            for i in range(SCALE_FACTOR):
                scaled_down = cv2.pyrDown(scaled_down)

            if frame_buffer is None:
                x, y, p = scaled_down.shape
                frame_buffer = np.ndarray((0, x, y, p))

            frame_buffer = np.append(frame_buffer, scaled_down[np.newaxis, ...], 0)

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    N, h, w, _channels = frame_buffer.shape

    processed_frame_buffer = np.ndarray((N, h, w, _channels))
    for x, y in np.ndindex(h, w):
        chunk = frame_buffer[:, x, y, :]
        processed_chunk = process_chunk(chunk, MAGNIFICATION_FACTOR, N)
        processed_frame_buffer[:, x, y, :] = processed_chunk

    cap = cv2.VideoCapture(abs_filepath)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            scaled_up = processed_frame_buffer[frame_count]

            for i in range(SCALE_FACTOR):
                scaled_up = cv2.pyrUp(scaled_up)

            cv2.imshow("Original", frame)
            cv2.imshow("Redone", scaled_up)

            frame_count = frame_count + 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break


if __name__ == "__main__":
    main()
