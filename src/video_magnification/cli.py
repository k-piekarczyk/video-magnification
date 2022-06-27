import cv2
import sys
import numpy as np
from scipy.fft import fft, fftfreq, ifft

from video_magnification.utils.video import (
    VideoFileReader,
    load_laplacian_pyramid_frames_to_buffers,
    merge_laplacian_pyramid_frames_into_single_buffer,
    load_frames_to_gaussian_buffer
)
from video_magnification.utils.math import get_max_pyramid_depth


def laplace():
    filepath = sys.argv[1]

    vfr = VideoFileReader(filepath=filepath)

    height, width, _ = vfr.get_stats()
    max_depth = get_max_pyramid_depth(height=height, width=width)
    DEPTH = max_depth

    pyramid_buffers, frame_count = load_laplacian_pyramid_frames_to_buffers(vfr=vfr, depth=DEPTH, color_space=cv2.COLOR_BGR2YCR_CB)
    # pyramid_buffers, frame_count = load_laplacian_pyramid_frames_to_buffers(vfr=vfr, depth=DEPTH)

    buffer = merge_laplacian_pyramid_frames_into_single_buffer(pyramid_buffers=pyramid_buffers, color_space=cv2.COLOR_YCR_CB2BGR)
    # buffer = merge_laplacian_pyramid_frames_into_single_buffer(pyramid_buffers=pyramid_buffers)

    stop = False
    while not stop:
        while not stop:
            for i in range(frame_count):

                cv2.imshow("Merged", buffer[i])

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop = True
                    break
    
def gauss(lower_frequency: float, higher_frequency: float, alpha: float):
    """
    Spatial filtering with Gaussian pyramid.

    Temporal filtering with ideal banpass filter
    """
    filepath = sys.argv[1]

    vfr = VideoFileReader(filepath=filepath)

    height, width, _, fps = vfr.get_stats()

    buffer, frame_count = load_frames_to_gaussian_buffer(vfr=vfr, depth=6, color_space=cv2.COLOR_BGR2YCR_CB)

    # cv2.imshow("channel 1", cv2.resize(buffer[0, :, :, 0], (width, height)))
    # cv2.imshow("channel 2", cv2.resize(buffer[0, :, :, 1], (width, height)))
    # cv2.imshow("channel 3", cv2.resize(buffer[0, :, :, 2], (width, height)))
    # cv2.waitKey()

    buffer = buffer.astype(np.float32) / 255

    n, h, w, c = buffer.shape

    frequencies = fftfreq(frame_count, 1/fps)
    bandpass = (frequencies >= lower_frequency) * (frequencies <= higher_frequency)

    processed_buffer = np.ndarray(shape=(n, h, w, c), dtype=buffer.dtype)
    for x, y in np.ndindex(h, w):
        chunk = buffer[:, x, y, :]
        processed_chunk = np.ndarray(shape=chunk.shape, dtype=chunk.dtype)

        ft_channel_1 = fft(chunk[:, 0])
        ft_channel_2 = fft(chunk[:, 1])
        ft_channel_3 = fft(chunk[:, 2])

        bandpassed_ft_channel_1 = [val if present else 0 for val, present in zip(ft_channel_1, bandpass)]
        bandpassed_ft_channel_2 = [val if present else 0 for val, present in zip(ft_channel_2, bandpass)]
        bandpassed_ft_channel_3 = [val if present else 0 for val, present in zip(ft_channel_3, bandpass)]

        processed_chunk[:, 0] = np.real(ifft(bandpassed_ft_channel_1)) * alpha
        processed_chunk[:, 1] = np.real(ifft(bandpassed_ft_channel_2)) * alpha
        processed_chunk[:, 2] = np.real(ifft(bandpassed_ft_channel_3)) * alpha

        processed_chunk = np.where(processed_chunk > 1, 1, processed_chunk)
        processed_chunk = np.where(processed_chunk < 0, 0, processed_chunk)

        processed_buffer[:, x, y, :] = processed_chunk

    
    cap = VideoFileReader(filepath=filepath).get_cap()

    out = cv2.VideoWriter('resources/output.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    
    i = 0
    while cap.isOpened():
        ret, original = cap.read()

        if ret:
            part = ((processed_buffer[i]) * 255).astype(np.uint8)
            part[:, :, 1] = part[:, :, 1] + 127
            part[:, :, 2] = part[:, :, 2] + 127

            frame = cv2.cvtColor(cv2.resize(part, (width, height)), cv2.COLOR_YCR_CB2BGR)

            # cv2.imshow("Merged", cv2.resize(cv2.cvtColor(processed_buffer[i], cv2.COLOR_YCR_CB2BGR)[:, :, 2], (width, height)))
            cv2.imshow("Original", original)
            cv2.imshow("Bandpassed", frame)
            cv2.imshow("Combined", cv2.add(original, frame))
            out.write(cv2.add(original, frame))

            # cv2.imshow("processed channel 1", , (width, height)))
            # cv2.imshow("processed channel 2", , (width, height)))
            # cv2.imshow("processed channel 3", , (width, height)))

            i += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop = True
                break
        else:
            break
    
    out.release()
    cv2.destroyAllWindows()

def main():
    # laplace()
    gauss(lower_frequency=80/60, higher_frequency=100/60, alpha=300)


if __name__ == "__main__":
    main()
