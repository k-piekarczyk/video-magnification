import numpy as np
from scipy.fft import fft, fftfreq, ifft


def process_chunk(chunk, magnification_factor: int, N) -> np.ndarray:

    b_channel = chunk[:, 0]
    g_channel = chunk[:, 1]
    r_channel = chunk[:, 2]

    normalized_b = [(x / 255) for x in b_channel]
    normalized_g = [(x / 255) for x in g_channel]
    normalized_r = [(x / 255) for x in r_channel]

    b_yf = fft(normalized_b)
    g_yf = fft(normalized_g)
    r_yf = fft(normalized_r)

    xf = fftfreq(N, 1 / 30)

    idx = (xf >= 1) * (xf <= 2)

    limited_b_yf = [val if det else 0 for val, det in zip(b_yf, idx)]
    limited_g_yf = [val if det else 0 for val, det in zip(g_yf, idx)]
    limited_r_yf = [val if det else 0 for val, det in zip(r_yf, idx)]

    ifft_b = ifft(limited_b_yf)
    ifft_g = ifft(limited_g_yf)
    ifft_r = ifft(limited_r_yf)

    real_ifft_b = np.real(ifft_b) * 256 #* magnification_factor
    real_ifft_g = np.real(ifft_g) * 256 #* magnification_factor
    real_ifft_r = np.real(ifft_r) * 256 #* magnification_factor

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

    processed_chunk = np.ndarray(shape=(real_ifft_g.shape[0], 3))
    processed_chunk[:, 0] = real_ifft_b
    processed_chunk[:, 1] = real_ifft_g
    processed_chunk[:, 2] = real_ifft_r

    return processed_chunk
