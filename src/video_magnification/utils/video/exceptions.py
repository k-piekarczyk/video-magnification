__all__ = ["NotAVideoFileException", "VideoNotOpeningException"]


class NotAVideoFileException(Exception):
    def __init__(self, filepath: str) -> None:
        super().__init__(f"Provided file '{filepath}' is not a video file")


class VideoNotOpeningException(Exception):
    def __init__(self) -> None:
        super().__init__("Couldn't open stream or file")


class ClosedVideoFileReaderException(Exception):
    def __init__(self) -> None:
        super().__init__("The VideoFileReader you're trying to access is closed")
