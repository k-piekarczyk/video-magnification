import cv2


__all__ = ["scale_frame_down", "scale_frame_up"]


def scale_frame_down(frame, scaling_factor: int):
    """
    Scales the given frame down using a Gaussian pyramid
    """
    for _ in range(scaling_factor):
        frame = cv2.pyrDown(frame)

    return frame


def scale_frame_up(frame, scaling_factor: int):
    """
    Scales the given frame down using a Gaussian pyramid
    """
    for _ in range(scaling_factor):
        frame = cv2.pyrUp(frame)

    return frame
