import cv2
import numpy as np
from typing import Optional
from scipy.fft import fft, fftfreq, ifft

from video_magnification.utils.video import VideoFileReader, load_frames_to_gaussian_buffer


__all__ = ["gauss_fft"]


def gauss_fft(
    filepath: str,
    depth: int,
    lower_frequency: float,
    higher_frequency: float,
    alpha: float,
    chroma_attenuation: float = 1.0,
    sampling_rate: Optional[int] = None,
    slice_pos: float = 0.5,
):
    """
    Spatial filtering: Gaussian pyramid
    Temporal filtering: ideal bandpass filter (FFT)
    """
    vfr = VideoFileReader(filepath=filepath)

    height, width, _, fps = vfr.get_stats()
    if sampling_rate is None:
        sampling_rate = fps

    buffer, frame_count = load_frames_to_gaussian_buffer(vfr=vfr, depth=depth, color_space=cv2.COLOR_BGR2YCR_CB)

    buffer = buffer.astype(np.float32) / 255
    buffer -= 0.5

    n, h, w, c = buffer.shape

    frequencies = fftfreq(frame_count, 1 / sampling_rate)
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

    # name_modifier = f"lf_{lower_frequency} hf_{higher_frequency} a_{alpha} chroma_{chroma_attenuation} sr_{sampling_rate}"

    out_combined = cv2.VideoWriter("resources/result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    out_comp = cv2.VideoWriter("resources/comparison.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 3, height))

    i = 0

    cv2.namedWindow("Comparison", cv2.WINDOW_NORMAL)

    window_width = width * 3
    window_height = height

    if (width * 3) > 1000:
        window_width = 1000
        window_height = int(height / ((width * 3) / 1000))

    cv2.resizeWindow("Comparison", window_width, window_height)

    org_slice = np.ndarray(shape=(height, n, c), dtype=np.uint8)
    motion_slice = np.ndarray(shape=(height, n, c), dtype=np.uint8)
    processed_slice = np.ndarray(shape=(height, n, c), dtype=np.uint8)

    while cap.isOpened():
        ret, original = cap.read()

        if ret:
            processed = ((processed_buffer[i]) * 255).astype(np.uint8)
            
            org_YCrCb = cv2.cvtColor(original, cv2.COLOR_BGR2YCR_CB)
            motion = cv2.resize(processed, (width, height))
            motion_BGR = cv2.cvtColor(motion, cv2.COLOR_YCR_CB2BGR)

            combined = cv2.add(org_YCrCb, motion)
            combined_BGR = cv2.cvtColor(combined, cv2.COLOR_YCR_CB2BGR)

            comp = cv2.hconcat([original, motion_BGR, combined_BGR])

            cv2.imshow("Comparison", comp)
            out_combined.write(combined_BGR)
            out_comp.write(comp)

            slice_index = int(width * slice_pos)
            org_slice[:, i] = original[:, slice_index]
            motion_slice[:, i] = motion_BGR[:, slice_index]
            processed_slice[:, i] = combined_BGR[:, slice_index]

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
