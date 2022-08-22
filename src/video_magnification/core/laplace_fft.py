import cv2
import numpy as np
import math

from typing import Optional
from scipy.fft import fft, fftfreq, ifft

from video_magnification.utils.video import (
    VideoFileReader,
    load_laplacian_pyramid_frames_to_buffers,
    merge_laplacian_pyramid_frames_into_single_buffer,
)
from video_magnification.utils.math import get_max_pyramid_depth


__all__ = ["laplace_fft"]


def laplace_fft(
    filepath: str,
    lower_frequency: float,
    higher_frequency: float,
    alpha: float,
    lambda_c: float,
    depth: Optional[int] = None,
    sampling_rate: Optional[int] = None,
    chroma_attenuation: float = 1.0,
    exageration: float = 1.0
):
    """
    Spatial filtering: Laplacian pyramid
    Temporal filtering: ideal bandpass filter (FFT)
    """
    vfr = VideoFileReader(filepath=filepath)

    height, width, _, fps = vfr.get_stats()
    if sampling_rate is None:
        sampling_rate = fps

    if depth is None:
        depth = get_max_pyramid_depth(height=height, width=width)

    pyramid_buffers, frame_count = load_laplacian_pyramid_frames_to_buffers(
        vfr=vfr, depth=depth, color_space=cv2.COLOR_BGR2YCR_CB
    )

    frequencies = fftfreq(frame_count, 1 / sampling_rate)
    bandpass = (frequencies >= lower_frequency) * (frequencies <= higher_frequency)

    levels = len(pyramid_buffers)

    delta = lambda_c/8/(1 + alpha)

    lmbda = math.sqrt(math.pow(height, 2) + math.pow(width, 2)) / 3

    processed_pyramid_buffers = []
    for level in reversed(range(levels)):
        buffer = pyramid_buffers[level]
        n, h, w, c = buffer.shape

        print(f"Processing level {level + 1} of {levels} [h: {h}, w: {w}] ...")

        current_alpha = (lmbda/delta/8 - 1)
        current_alpha *= exageration

        if current_alpha > alpha:
            current_alpha = alpha

        buffer = buffer.astype(np.float32) / 255

        processed_buffer = np.ndarray(shape=(n, h, w, c), dtype=buffer.dtype)
        if not (level == 0 or level == levels - 1):
            for x, y in np.ndindex(h, w):
                chunk = buffer[:, x, y, :]
                processed_chunk = np.ndarray(shape=chunk.shape, dtype=chunk.dtype)

                if chunk[:, 0].max() > 0:
                    ft_channel_1 = fft(chunk[:, 0])
                    bandpassed_ft_channel_1 = [val if present else 0 for val, present in zip(ft_channel_1, bandpass)]
                    processed_chunk[:, 0] = np.real(ifft(bandpassed_ft_channel_1)) * current_alpha
                else:
                    processed_chunk[:, 0] = 0

                if chunk[:, 1].max() > 0 and chroma_attenuation > 0:
                    ft_channel_2 = fft(chunk[:, 1])
                    bandpassed_ft_channel_2 = [val if present else 0 for val, present in zip(ft_channel_2, bandpass)]
                    processed_chunk[:, 1] = np.real(ifft(bandpassed_ft_channel_2)) * current_alpha * chroma_attenuation
                else:
                    processed_chunk[:, 1] = 0

                if chunk[:, 2].max() > 0 and chroma_attenuation > 0:
                    ft_channel_3 = fft(chunk[:, 2])
                    bandpassed_ft_channel_3 = [val if present else 0 for val, present in zip(ft_channel_3, bandpass)]
                    processed_chunk[:, 2] = np.real(ifft(bandpassed_ft_channel_3)) * current_alpha * chroma_attenuation
                else:
                    processed_chunk[:, 2] = 0

                processed_chunk = np.where(processed_chunk > 1, 1, processed_chunk)
                processed_chunk = np.where(processed_chunk < 0, 0, processed_chunk)

                processed_buffer[:, x, y, :] = processed_chunk
        else:
            processed_buffer[:, :, :, :] = 0

        processed = ((processed_buffer) * 255).astype(np.uint8)
        processed_pyramid_buffers.append(processed)

        lmbda /= 2

    processed_pyramid_buffers.reverse()
    merged_buffer = merge_laplacian_pyramid_frames_into_single_buffer(pyramid_buffers=processed_pyramid_buffers)[:, :height, :width, :]

    print("Saving results...")

    cap = VideoFileReader(filepath=filepath).get_cap()
    out_combined = cv2.VideoWriter("resources/result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    out_comp = cv2.VideoWriter("resources/comparison.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 3, height))

    i = 0
    while cap.isOpened():
        ret, original = cap.read()

        if ret:
            org_YCrCb = cv2.cvtColor(original, cv2.COLOR_BGR2YCR_CB)
            motion = merged_buffer[i]
            motion_BGR = cv2.cvtColor(motion, cv2.COLOR_YCR_CB2BGR)

            combined = cv2.add(org_YCrCb, motion)
            combined_BGR = cv2.cvtColor(combined, cv2.COLOR_YCR_CB2BGR)

            comp = cv2.hconcat([original, motion_BGR, combined_BGR])

            out_combined.write(combined_BGR)
            out_comp.write(comp)

            i += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    out_combined.release()
    out_comp.release()
