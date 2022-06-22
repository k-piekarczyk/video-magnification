import numpy as np
import numpy.typing as npt
from scipy.fft import fft, fftfreq, ifft


def process_frame_buffer(frame_buffer: npt.NDArray[np.uint8], magnification_factor: int = 1):
    N, h, w, _channels = frame_buffer.shape

    processed_frame_buffer: npt.NDArray[np.uint8] = np.ndarray((N, h, w, _channels))
    for x, y in np.ndindex(h, w):
        chunk = frame_buffer[:, x, y, :]
        processed_chunk = process_chunk(chunk=chunk, magnification_factor=magnification_factor, N=N)
        processed_frame_buffer[:, x, y, :] = processed_chunk

    return processed_frame_buffer


def process_chunk(chunk, magnification_factor: int, N) -> npt.NDArray:

    xf = fftfreq(N, 1 / 30)
    idx = (xf >= 1) * (xf <= 2)

    b_channel = chunk[:, 0]
    g_channel = chunk[:, 1]
    r_channel = chunk[:, 2]

    normalized_b = [(x / 255) for x in b_channel]
    normalized_g = [(x / 255) for x in g_channel]
    normalized_r = [(x / 255) for x in r_channel]

    b_yf = fft(normalized_b)
    g_yf = fft(normalized_g)
    r_yf = fft(normalized_r)

    limited_b_yf = [val if det else 0 for val, det in zip(b_yf, idx)]
    limited_g_yf = [val if det else 0 for val, det in zip(g_yf, idx)]
    limited_r_yf = [val if det else 0 for val, det in zip(r_yf, idx)]

    ifft_b = ifft(limited_b_yf)
    ifft_g = ifft(limited_g_yf)
    ifft_r = ifft(limited_r_yf)

    real_ifft_b = np.real(ifft_b) * 256 * magnification_factor
    real_ifft_g = np.real(ifft_g) * 256 * magnification_factor
    real_ifft_r = np.real(ifft_r) * 256 * magnification_factor

    # print(type(real_ifft_g), real_ifft_g.shape)
    #
    # b_r_diff = real_ifft_b - real_ifft_r
    #
    # peaks, _ = find_peaks(b_r_diff, height=0)
    #
    # wave_length_fr = np.average(np.diff(peaks))
    # wave_length_s = wave_length_fr / 30
    #
    # bpm = (1 / wave_length_s) * 60

    processed_chunk = np.ndarray(shape=(real_ifft_g.shape[0], 3)).astype(np.uint8)
    processed_chunk[:, 0] = real_ifft_b
    processed_chunk[:, 1] = real_ifft_g
    processed_chunk[:, 2] = real_ifft_r

    return processed_chunk
