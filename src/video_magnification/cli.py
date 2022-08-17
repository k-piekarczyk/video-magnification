import sys
import time
from video_magnification.core import gauss_fft, laplace_fft


def run(filepath: str):
    start_time = time.time()

    # my_face
    # laplace_fft(filepath, lower_frequency=75/60, higher_frequency=90/60, alpha=100, lambda_c=100)
    # gauss_fft(filepath, 1, lower_frequency=75/60, higher_frequency=90/60, alpha=50)

    # face
    # laplace_fft(filepath, lower_frequency=50/60, higher_frequency=60/60, alpha=100, lambda_c=20)
    # gauss_fft(filepath, 5, lower_frequency=50/60, higher_frequency=60/60, alpha=100)

    # wrist
    laplace_fft(filepath, lower_frequency=50/60, higher_frequency=60/60, alpha=50, lambda_c=3)
    # gauss_fft(filepath, 0, lower_frequency=50/60, higher_frequency=60/60, alpha=50)

    # guitar E
    # laplace_fft(filepath, lower_frequency=72, higher_frequency=92, alpha=50, lambda_c=10, chroma_attenuation=0, sampling_rate=600)

    # Guitar A
    # laplace_fft(filepath, lower_frequency=100, higher_frequency=120, alpha=50, chroma_attenuation=1, sampling_rate=600)

    # gauss_fft(
    #     filepath,
    #     0,
    #     lower_frequency=100,
    #     higher_frequency=120,
    #     alpha=50,
    #     chroma_attenuation=0,
    #     sampling_rate=600,
    #     slice_pos=2 / 3,
    # )

    # Bridge 1
    # laplace_fft(filepath, lower_frequency=.5, higher_frequency=2, alpha=50)
    elapsed = time.time() - start_time

    print(f"This took {elapsed} seconds!")

def cli():
    filepath = sys.argv[1]
    run(filepath)


if __name__ == "__main__":
    filepath = sys.argv[1]
    run(filepath)
