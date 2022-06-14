__all__ = ["NotAVideoFileException", "VideoNotOpeningException"]


class NotAVideoFileException(Exception):
    def __init__(self, filepath: str) -> None:
        super(NotAVideoFileException).__init__(f"Provided file '{filepath}' is not a video file")


class VideoNotOpeningException(Exception):
    def __init__(self) -> None:
        super(VideoNotOpeningException).__init__("Couldn't open stream or file")


class ClosedVideoFileReaderException(Exception):
    def __init__(self) -> None:
        super(ClosedVideoFileReaderException).__init__("The VideoFileReader you're trying to access is closed")
