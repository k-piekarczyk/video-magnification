import cv2
import sys
import numpy.typing as npt
import numpy as np

from video_magnification.processing import process_frame_buffer
from video_magnification.utils.video import (
    VideoFileReader,
    scale_frame_up,
    laplace_pyramid_step,
    prepare_for_laplace_pyramid,
)

SCALE_FACTOR = 3
MAGNIFICATION_FACTOR = 4


def legacy():
    filepath = sys.argv[1]

    vf_reader = VideoFileReader(filepath=filepath)

    frame_buffer = vf_reader.load_frames_into_buffer(SCALE_FACTOR)

    processed_frame_buffer = process_frame_buffer(frame_buffer=frame_buffer, magnification_factor=MAGNIFICATION_FACTOR)

    cap = VideoFileReader(filepath=filepath).get_cap()
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            scaled_up = scale_frame_up(frame=processed_frame_buffer[frame_count], scaling_factor=SCALE_FACTOR)[
                :-6, :, :
            ].astype("uint8")

            cv2.imshow("Original", frame)
            cv2.imshow("Redone", scaled_up)

            alpha = 0.95
            beta = 1 - alpha

            cv2.imshow("Joined", cv2.addWeighted(frame, alpha, scaled_up, beta, 0.0))

            frame_count = frame_count + 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break


def laplace():
    filepath = sys.argv[1]

    cap = VideoFileReader(filepath=filepath).get_cap()

    while True:
        stop = False
        cap = VideoFileReader(filepath=filepath).get_cap()

        while cap.isOpened():
            ret, raw_frame = cap.read()
            raw_frame: npt.NDArray[np.uint8]  # just here for type annotation

            if ret:
                # original_shape = raw_frame.shape

                frame, max_depth = prepare_for_laplace_pyramid(frame=raw_frame)

                # frame = cv2.cvtColor(prep_frame, code=cv2.COLOR_BGR2YCR_CB)[:, :, 0:1]

                cv2.imshow("Original", frame)

                s, d = laplace_pyramid_step(frame=frame)
                scaled = [s]
                diff = [d]

                DEPTH = max_depth
                for i in range(1, DEPTH):
                    s, d = laplace_pyramid_step(frame=scaled[i - 1])
                    scaled.append(s)
                    diff.append(d)

                for i in range(DEPTH):
                    cv2.imshow(f"Scaled: {i+1}", scaled[i])
                    if scaled[i].shape[1] < 300:
                        cv2.resizeWindow(f"Scaled: {i+1}", 300, scaled[i].shape[0])

                    cv2.imshow(f"Diff: {i+1}", diff[i])
                    if diff[i].shape[1] < 300:
                        cv2.resizeWindow(f"Diff: {i+1}", 300, diff[i].shape[0])

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop = True
                    break
            else:
                break
        if stop:
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()


def main():
    # legacy()
    laplace()


if __name__ == "__main__":
    main()
