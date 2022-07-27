from typing import Optional
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


def laplace(lower_frequency: float, higher_frequency: float, alpha: float, depth: Optional[int] = None):
    filepath = sys.argv[1]

    vfr = VideoFileReader(filepath=filepath)

    height, width, _, fps = vfr.get_stats()

    if depth is None:
        depth = get_max_pyramid_depth(height=height, width=width)

    pyramid_buffers, frame_count = load_laplacian_pyramid_frames_to_buffers(vfr=vfr, depth=depth, color_space=cv2.COLOR_BGR2YCR_CB)

    levels = len(pyramid_buffers)

    processed_pyramid_buffers = []
    for level in range(levels):
        buffer = pyramid_buffers[level]
        n, h, w, c = buffer.shape
        
        print(f"Processing level {level + 1} of {levels} [h: {h}, w: {w}] ...")

        buffer = buffer.astype(np.float32) / 255

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

        processed = ((processed_buffer) * 255).astype(np.uint8)
        # processed[:, :, 1] = processed[:, :, 1] + 127
        # processed[:, :, 2] = processed[:, :, 2] + 127
        processed_pyramid_buffers.append(processed)
        
    merged_buffer = merge_laplacian_pyramid_frames_into_single_buffer(pyramid_buffers=processed_pyramid_buffers)[:, :height, :width, :]

    cap = VideoFileReader(filepath=filepath).get_cap()
    out_combined = cv2.VideoWriter('resources/output.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    out_comp = cv2.VideoWriter('resources/output_motion.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    
    i = 0
    while cap.isOpened():
        ret, original = cap.read()

        if ret:
            motion = cv2.cvtColor(merged_buffer[i], cv2.COLOR_YCR_CB2BGR)

            cv2.imshow("Original", original)
            cv2.imshow("Bandpassed", motion)
            cv2.imshow("Combined", cv2.add(original, motion))
            out_combined.write(cv2.add(original, motion))
            out_comp.write(motion)

            i += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    
    out_combined.release()
    out_comp.release()
    cv2.destroyAllWindows()
    
def gauss(depth: int, lower_frequency: float, higher_frequency: float, alpha: float, chroma_attenuation: Optional[float] = 1.0):
    """
    Spatial filtering with Gaussian pyramid.

    Temporal filtering with ideal banpass filter
    """
    filepath = sys.argv[1]

    vfr = VideoFileReader(filepath=filepath)

    height, width, _, fps = vfr.get_stats()

    buffer, frame_count = load_frames_to_gaussian_buffer(vfr=vfr, depth=depth, color_space=cv2.COLOR_BGR2YCR_CB)

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
        processed_chunk[:, 1] = np.real(ifft(bandpassed_ft_channel_2)) * alpha * chroma_attenuation
        processed_chunk[:, 2] = np.real(ifft(bandpassed_ft_channel_3)) * alpha * chroma_attenuation

        processed_chunk = np.where(processed_chunk > 1, 1, processed_chunk)
        processed_chunk = np.where(processed_chunk < 0, 0, processed_chunk)

        processed_buffer[:, x, y, :] = processed_chunk

    
    cap = VideoFileReader(filepath=filepath).get_cap()

    out_combined = cv2.VideoWriter('resources/result.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    out_comp = cv2.VideoWriter('resources/comparison.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 3,height))
    
    i = 0

    cv2.namedWindow('Comparison', cv2.WINDOW_NORMAL)

    window_width = width * 3
    window_height = height

    if (width * 3) > 1000:
        window_width = 1000
        window_height = int(height / ((width * 3)/1000))
    
    cv2.resizeWindow('Comparison', window_width, window_height)

    org_slice = np.ndarray(shape=(height, n, c), dtype=np.uint8)
    motion_slice = np.ndarray(shape=(height, n, c), dtype=np.uint8)
    processed_slice = np.ndarray(shape=(height, n, c), dtype=np.uint8)

    while cap.isOpened():
        ret, original = cap.read()

        if ret:
            processed = ((processed_buffer[i]) * 255).astype(np.uint8)
            processed[:, :, 1] = processed[:, :, 1] + 127
            processed[:, :, 2] = processed[:, :, 2] + 127

            motion = cv2.cvtColor(cv2.resize(processed, (width, height)), cv2.COLOR_YCR_CB2BGR)
            combined = cv2.add(original, motion)
            comp = cv2.hconcat([original, motion, combined])

            cv2.imshow("Comparison", comp)
            out_combined.write(combined)
            out_comp.write(comp)

            slice_index = int(width / 2)
            org_slice[:, i] = original[:, slice_index]
            motion_slice[:, i] = motion[:, slice_index]
            processed_slice[:, i] = combined[:, slice_index]

            i += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    
    out_combined.release()
    out_comp.release()
    cv2.destroyAllWindows()

    cv2.imshow("Original slice", org_slice)
    cv2.imshow("Motion slice", motion_slice)
    cv2.imshow("Processed slice", processed_slice)
    cv2.waitKey()
    cv2.destroyAllWindows()

def main():
    # laplace(lower_frequency=10, higher_frequency=20, alpha=10)
    gauss(5, lower_frequency=50/60, higher_frequency=60/60, alpha=50, chroma_attenuation=1)


if __name__ == "__main__":
    main()
