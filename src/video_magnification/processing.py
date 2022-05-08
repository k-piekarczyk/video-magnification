import cv2
import numpy as np
from scipy.fft import fft, fftfreq, ifft
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import sys
import os.path


def process_chunk(chunk, magnification_factor: int, N):

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

    idx = (xf > 1) * (xf < 2)

    limited_b_yf = [val if det else 0 for val, det in zip(b_yf, idx)]
    limited_g_yf = [val if det else 0 for val, det in zip(g_yf, idx)]
    limited_r_yf = [val if det else 0 for val, det in zip(r_yf, idx)]

    ifft_b = ifft(limited_b_yf)
    ifft_g = ifft(limited_g_yf)
    ifft_r = ifft(limited_r_yf)

    real_ifft_b = np.real(ifft_b) * 256 * magnification_factor
    real_ifft_g = np.real(ifft_g) * 256 * magnification_factor
    real_ifft_r = np.real(ifft_r) * 256 * magnification_factor

    # figure, axis = plt.subplots(4)
    #
    # axis[0].plot(b_channel, "b-")
    # axis[1].stem(xf, np.abs(limited_b_yf), linefmt="b")
    # axis[2].plot(real_ifft_b, "b-")
    #
    # axis[0].plot(range(N), g_channel, "g-")
    # axis[1].stem(xf, np.abs(limited_g_yf), linefmt="g")
    # axis[2].plot(real_ifft_g, "g-")
    #
    # axis[0].plot(range(N), r_channel, "r-")
    # axis[1].stem(xf, np.abs(limited_r_yf), linefmt="r")
    # axis[2].plot(real_ifft_r, "r-")

    b_r_diff = np.real(ifft_g) - np.real(ifft_r)
    #
    # axis[3].plot(b_r_diff)
    #
    peaks, _ = find_peaks(b_r_diff, height=0)
    #
    # axis[3].plot(peaks, b_r_diff[peaks], "x")

    wave_length_fr = np.average(np.diff(peaks))
    wave_length_s = wave_length_fr / 30

    hr = 1 / wave_length_s

    print(f"Heart rate: {hr * 60} BPM")

    plt.show()
