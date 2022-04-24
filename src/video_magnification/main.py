import cv2
import numpy as np
from scipy.fft import fft, fftfreq, ifft
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import sys
import os.path

def main(filepath: str):
    abs_filepath = os.path.abspath(filepath)
    if not os.path.isfile(abs_filepath):
        raise Exception("Provided filepath should be to a movie")

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
            rows, cols, _channels = map(int, frame.shape)

            scaled_down = frame

            for i in range(6):
                scaled_down = cv2.pyrDown(scaled_down)

            if frame_buffer is None:
                x, y, p = scaled_down.shape
                frame_buffer = np.ndarray((0, x, y, p))

            frame_buffer = np.append(frame_buffer, scaled_down[np.newaxis, ...], 0)
            # print(frame_buffer.shape)

            # cv2.imshow('Frame', frame)
            # cv2.imshow('Scaled Down', scaled_down)

            # # Press Q on keyboard to  exit
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    N, h, w, _channels = frame_buffer.shape

    coord = (int(h / 2), int(w / 2))

    b_channel = frame_buffer[:, coord[0], coord[1], 0]
    g_channel = frame_buffer[:, coord[0], coord[1], 1]
    r_channel = frame_buffer[:, coord[0], coord[1], 2]

    org_upper = 255
    lower, upper = -1, 1
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

    real_ifft_b = np.real(ifft_b) * 256
    real_ifft_g = np.real(ifft_g) * 256
    real_ifft_r = np.real(ifft_r) * 256

    figure, axis = plt.subplots(4)

    axis[0].plot(b_channel, "b-")
    axis[1].stem(xf, np.abs(limited_b_yf), linefmt="b")
    axis[2].plot(np.real(ifft_b) * 255, "b-")

    axis[0].plot(range(N), g_channel, "g-")
    axis[1].stem(xf, np.abs(limited_g_yf), linefmt="g")
    axis[2].plot(np.real(ifft_g) * 255, "g-")

    axis[0].plot(range(N), r_channel, "r-")
    axis[1].stem(xf, np.abs(limited_r_yf), linefmt="r")
    axis[2].plot(np.real(ifft_r) * 255, "r-")

    b_r_diff = np.real(ifft_g) - np.real(ifft_r)

    axis[3].plot(b_r_diff)

    peaks, _ = find_peaks(b_r_diff, height=0)

    axis[3].plot(peaks, b_r_diff[peaks], "x")

    wave_length_fr = np.average(np.diff(peaks))
    wave_length_s = wave_length_fr / 30

    hr = 1 / wave_length_s

    print(f"Heart rate: {hr * 60} BPM")

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1])
