#
#            PySceneDetect: Python-Based Video Scene Detector
#   -------------------------------------------------------------------
#     [  Site:    https://scenedetect.com                           ]
#     [  Docs:    https://scenedetect.com/docs/                     ]
#     [  Github:  https://github.com/Breakthrough/PySceneDetect/    ]
#
# Copyright (C) 2022 Brandon Castellano <http://www.bcastell.com>.
# PySceneDetect is licensed under the BSD 3-Clause License; see the
# included LICENSE file, or visit one of the above pages for details.
#
""":class:`VideoStreamCv2` is backed by the OpenCV `VideoCapture` object. This is the default
backend. Works with video files, image sequences, and network streams/URLs.

For wrapping input devices or pipes, there is also :class:`VideoCaptureAdapter` which can be
constructed from an existing `cv2.VideoCapture`. This allows performing scene detection on inputs
which do not support seeking.
"""

import math
import os
import os.path
import warnings
from fractions import Fraction
from logging import getLogger

import cv2
import numpy as np

from scenedetect.common import (
    MAX_FPS_DELTA,
    FrameRate,
    FrameTimecode,
    Timecode,
    TimecodeLike,
    framerate_to_fraction,
)
from scenedetect.platform import StrPath, get_file_name
from scenedetect.video_stream import (
    FrameRateUnavailable,
    SeekError,
    VideoOpenFailure,
    VideoStream,
)

logger = getLogger("pyscenedetect")

IMAGE_SEQUENCE_IDENTIFIER = "%"

NON_VIDEO_FILE_INPUT_IDENTIFIERS = (
    IMAGE_SEQUENCE_IDENTIFIER,  # image sequence
    "://",  # URL/network stream
    " ! ",  # gstreamer pipe
)

# Tick-rate frame counters (where CAP_PROP_FPS reports a time-base tick rate such as 600 Hz
# instead of the real frame rate) are produced by ffmpeg when remuxing into AVI. The cadence
# probe below is therefore restricted to AVI containers: other containers report a usable FPS
# directly, and probing files whose leading frames lack timestamps (e.g. MPEG-TS with missing
# PTS, see issue #590) would derive a bogus rate from sparse timestamps.
AVI_EXTENSIONS = (".avi",)


def _is_avi_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in AVI_EXTENSIONS


def _get_capture_api(path: str) -> int:
    """API preference to use when opening `path` with `cv2.VideoCapture`. This exists as a
    workaround for https://scenedetect.com/issue/575, where certain versions of the opencv-python
    package cause intermittent segfaults on exit. The issue only happens on Windows when opening
    PNG image sequences.

    By default OpenCV resolves any capture backend by probing for compatibility, and tries FFmpeg
    first. Usually this fails, and it falls back gracefully to CAP_IMAGES. However, this also seems
    to cause issues on interpreter exit, as the stack trace implicates something in the ffmpeg DLL.

    TODO: Remove this workaround once upstream fix ships.

    https://github.com/opencv/opencv/issues/30017
    """
    is_local = not any(identifier in path for identifier in ("://", " ! "))
    if os.name == "nt" and is_local and path.lower().endswith(".png"):
        return cv2.CAP_IMAGES
    return cv2.CAP_ANY


def _get_aspect_ratio(cap: cv2.VideoCapture, epsilon: float = 0.0001) -> float:
    """Display/pixel aspect ratio of the VideoCapture as a float (1.0 represents square pixels)."""
    # Versions of OpenCV < 3.4.1 do not support this, so we fall back to 1.0.
    if "CAP_PROP_SAR_NUM" not in dir(cv2):
        return 1.0
    num: float = cap.get(cv2.CAP_PROP_SAR_NUM)
    den: float = cap.get(cv2.CAP_PROP_SAR_DEN)
    # If numerator or denominator are close to zero, so we fall back to 1.0.
    if abs(num) < epsilon or abs(den) < epsilon:
        return 1.0
    return num / den


# Maximum number of frames scanned by `_probe_frame_cadence`. Frames whose timestamp OpenCV cannot
# establish yet (e.g. the leading packets of an MPEG-TS with missing PTS) report POS_MSEC of zero,
# so the probe keeps grabbing until it sees two timestamps or reaches this bound.
_MAX_CADENCE_PROBE_FRAMES = 16


def _probe_frame_cadence(cap: cv2.VideoCapture) -> float | None:
    """Measure the real frame rate from frame presentation timestamps.

    Some containers (notably AVI files remuxed by ffmpeg) expose a high *tick rate* through
    `CAP_PROP_FPS` (e.g. 600 Hz for 23.976 fps content) while `CAP_PROP_POS_MSEC` still follows
    the true frame cadence. This returns the rate derived from the first two frame timestamps,
    or ``None`` if no usable timestamps are available (the caller keeps the reported FPS). The
    capture position is left wherever the last `grab()` landed; callers must rewind afterwards.
    """
    first_ms: float | None = None
    for _ in range(_MAX_CADENCE_PROBE_FRAMES):
        if not cap.grab():
            break
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if ms <= 0.0:
            continue
        if first_ms is None:
            first_ms = ms
            continue
        elapsed_ms = ms - first_ms
        # Two timestamps separated by `elapsed_ms` span one frame interval (the frames at and
        # after the first timestamp). Ignore non-positive deltas (non-monotonic timestamps).
        if elapsed_ms > 0.0:
            return 1000.0 / elapsed_ms
    return None


class VideoStreamCv2(VideoStream):
    """OpenCV `cv2.VideoCapture` backend."""

    def __init__(
        self,
        path: StrPath | None = None,
        frame_rate: FrameRate | None = None,
        max_decode_attempts: int = 5,
        path_or_device: StrPath | int | None = None,
        framerate: float | None = None,
    ):
        """Open a video file, image sequence, or network stream.

        Arguments:
            path: Path to the video. Can be a file, image sequence (`'folder/DSC_%04d.jpg'`),
                or network stream.
            frame_rate: If set, overrides the detected frame rate. Takes precedence over
                `framerate`.
            max_decode_attempts: Number of attempts to continue decoding the video
                after a frame fails to decode. This allows processing videos that
                have a few corrupted frames or metadata (in which case accuracy
                of detection algorithms may be lower). Once this limit is passed,
                decoding will stop and emit an error.
            path_or_device: [DEPRECATED] Specify `path` for files, image sequences, or
                network streams/URLs.  Use `VideoCaptureAdapter` for devices/pipes.
            framerate: [DEPRECATED] Use `frame_rate` instead. Retained as a deprecated
                alias for backwards compatibility; ignored when `frame_rate` is provided.

        Raises:
            OSError: file could not be found or access was denied
            VideoOpenFailure: video could not be opened (may be corrupted)
            ValueError: specified frame rate is invalid
        """
        super().__init__()
        if framerate is not None:
            warnings.warn(
                "`framerate` is deprecated and scheduled for removal in v0.9; "
                "use `frame_rate` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if frame_rate is None:
            frame_rate = framerate
        if path_or_device is not None:
            warnings.warn(
                "The `path_or_device` argument is deprecated, use `path` or `VideoCaptureAdapter`"
                " instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            resolved: str | int = (
                path_or_device if isinstance(path_or_device, int) else os.fspath(path_or_device)
            )
        elif path is None:
            raise ValueError("Path must be specified!")
        else:
            resolved = os.fspath(path)
        if frame_rate is not None and frame_rate < MAX_FPS_DELTA:
            raise ValueError(f"Specified frame rate ({float(frame_rate):f}) is invalid!")
        if max_decode_attempts < 0:
            raise ValueError("Maximum decode attempts must be >= 0!")

        self._path_or_device: str | int = resolved
        self._is_device = isinstance(self._path_or_device, int)

        # VideoCapture state
        self._has_grabbed = False
        self._max_decode_attempts = max_decode_attempts
        self._decode_failures = 0
        self._warning_displayed = False
        # POS_MSEC (microseconds) reported for the first decoded frame, subtracted from every
        # timestamp so the first frame is anchored at time 0 like the other backends. Decided
        # lazily on the first sequential `read()` because some containers only establish
        # timestamps then. Only meaningful while reading from the start: a seek lands on an
        # arbitrary timestamp, so the origin must not be (re)captured there.
        self._time_origin_us: int | None = None
        # True while frames are read sequentially from the start of the stream. A seek jumps to
        # an arbitrary timestamp and flips this off, preventing the seek position from being
        # mistaken for the first-frame time origin.
        self._sequential = True
        # True after a seek on a tick-rate container. OpenCV counts real frames in POS_FRAMES
        # during a sequential read, but reports *ticks* there after a seek, so `frame_number`
        # must convert until the next sequential pass (seek(0)/reset) begins.
        self._post_seek_tick_position = False
        # Anchor for post-seek frame accounting on a tick-rate container. `_seek_anchor_ticks`
        # is the raw POS_FRAMES value at the seek landing point and `_seek_anchor_frame` the
        # 0-based frame the next read() returns minus one; afterwards each grab advances the tick
        # counter and the real frame number by one, so real frames follow as
        # anchor_frame + (raw_ticks - anchor_ticks).
        self._seek_anchor_ticks: float | None = None
        self._seek_anchor_frame: int | None = None
        # Frame count corrected from a tick-rate counter (e.g. remuxed AVI); None means the raw
        # CAP_PROP_FRAME_COUNT is used. Cached so `duration` stays stable after `close()`.
        self._corrected_frame_count: int | None = None
        # Whether the caller explicitly set a frame rate (disables the tick-rate probe and is
        # preserved across `reset()`).
        self._frame_rate_overridden = frame_rate is not None

        # `_open_capture` populates `_cap` and `_frame_rate`.
        self._open_capture(frame_rate)

    #
    # Backend-Specific Methods/Properties
    #

    @property
    def capture(self) -> cv2.VideoCapture:
        """Returns reference to underlying VideoCapture object. Use with caution.

        Prefer to use this property only to take ownership of the underlying cv2.VideoCapture object
        backing this object. Seeking or using the read/grab methods through this property are
        unsupported and will leave this object in an inconsistent state. Closing this object (e.g.
        by leaving a `with` block) releases the capture.
        """
        return self._cap

    #
    # VideoStream Methods/Properties
    #

    BACKEND_NAME = "opencv"
    """Unique name used to identify this backend."""

    @property
    def frame_rate(self) -> Fraction:
        return self._frame_rate

    @property
    def path(self) -> str:
        if self._is_device:
            assert isinstance(self._path_or_device, int)
            return f"Device {self._path_or_device}"
        assert isinstance(self._path_or_device, str)
        return self._path_or_device

    @property
    def name(self) -> str:
        if self._is_device:
            return self.path
        file_name: str = get_file_name(self.path, include_extension=False)
        if IMAGE_SEQUENCE_IDENTIFIER in file_name:
            # file_name is an image sequence, trim everything including/after the %.
            # TODO: This excludes any suffix after the sequence identifier.
            file_name = file_name[: file_name.rfind(IMAGE_SEQUENCE_IDENTIFIER)]
        return file_name

    @property
    def is_seekable(self) -> bool:
        """True if seek() is allowed, False otherwise.

        Always False if opening a device/webcam."""
        return not self._is_device

    @property
    def frame_size(self) -> tuple[int, int]:
        """Size of each video frame in pixels as a tuple of (width, height)."""
        if self._closed:
            return self._closed_frame_size
        return (
            math.trunc(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            math.trunc(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def duration(self) -> FrameTimecode | None:
        """Duration of the stream as a FrameTimecode, or None if non terminating."""
        if self._is_device:
            return None
        if self._corrected_frame_count is not None:
            return self.base_timecode + self._corrected_frame_count
        if self._closed:
            return self.base_timecode + math.trunc(self._closed_frame_count)
        return self.base_timecode + math.trunc(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def aspect_ratio(self) -> float:
        """Display/pixel aspect ratio as a float (1.0 represents square pixels)."""
        if self._closed:
            return self._closed_aspect_ratio
        return _get_aspect_ratio(self._cap)

    @property
    def timecode(self) -> Timecode:
        """Current position within stream as a Timecode."""
        # *NOTE*: Although OpenCV has `CAP_PROP_PTS`, it doesn't seem to be reliable. For now, we
        # use `CAP_PROP_POS_MSEC` instead, converting to microseconds for sufficient precision to
        # avoid frame-boundary rounding errors at common framerates like 24000/1001.
        if self._eof_timecode is not None:
            return self._eof_timecode
        # Tick-rate containers (e.g. the remuxed AVI) expose POS_MSEC at the tick rate and reset
        # it to zero near end of stream, so derive the presentation time from the real frame
        # number at the actual frame rate instead of trusting the raw timestamp.
        if self._capture_rate != self._frame_rate and self._has_grabbed:
            return self._frame_number_timecode()
        ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        time_base = Fraction(1, 1000000)
        pts = round(ms * 1000)
        # Anchor the first decoded frame at time 0. Some containers (e.g. DTS-only AVI) report a
        # nonzero timestamp for frame 1; subtracting it keeps first/last timecodes consistent with
        # the PyAV and MoviePy backends.
        if self._time_origin_us is not None:
            pts -= self._time_origin_us
        return Timecode(pts=pts, time_base=time_base)

    def _frame_number_timecode(self) -> Timecode:
        """Synthesize the presentation Timecode of the current frame from its real frame number
        and the stream frame rate. Used for tick-rate containers whose raw POS_MSEC does not
        reflect the frame cadence (and can reset to zero near end of stream)."""
        fps = self._frame_rate
        return Timecode(
            pts=max(self.frame_number - 1, 0) * fps.denominator,
            time_base=Fraction(1, fps.numerator),
        )

    @property
    def position(self) -> FrameTimecode:
        timecode = self.timecode
        # If PTS is non-positive but we've read frames, derive from frame number. This handles
        # image sequences and cases where CAP_PROP_POS_MSEC is unreliable. OpenCV 5 reports
        # CAP_PROP_POS_MSEC as -1 (rather than 0) for image sequences on Windows, so check <= 0.
        if timecode.pts <= 0 and self.frame_number > 0:
            fps = self.frame_rate
            time_base = Fraction(1, fps.numerator)
            pts = (self.frame_number - 1) * fps.denominator
            timecode = Timecode(pts=pts, time_base=time_base)
        return FrameTimecode(timecode=timecode, fps=self.frame_rate)

    @property
    def position_ms(self) -> float:
        # Tick-rate containers derive their position from the real frame number (see `timecode`),
        # since the raw POS_MSEC uses the tick time base and can reset near end of stream.
        if self._capture_rate != self._frame_rate and self._has_grabbed:
            return max(self.frame_number - 1, 0) * 1000.0 / float(self._frame_rate)
        ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        # Apply the same first-frame offset as `timecode` so the two stay consistent.
        if self._time_origin_us is not None:
            return max(0.0, ms - self._time_origin_us / 1000.0)
        return ms

    @property
    def frame_number(self) -> int:
        # POS_FRAMES counts actual decoded frames during a sequential read, even for tick-rate
        # containers (only POS_MSEC and CAP_PROP_FPS use the tick time base). After a seek the
        # counter still increments one per grab but its value is indexed in *ticks*, so recover
        # the real frame number relative to the anchor recorded at the seek landing point.
        raw = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        if self._post_seek_tick_position:
            if self._seek_anchor_ticks is not None and self._seek_anchor_frame is not None:
                frame_num = self._seek_anchor_frame + math.trunc(
                    raw - self._seek_anchor_ticks
                )
            else:
                ticks_per_frame = float(self._capture_rate) / float(self._frame_rate)
                frame_num = math.ceil(raw / ticks_per_frame - 1e-9)
            if self._corrected_frame_count is not None:
                frame_num = min(frame_num, self._corrected_frame_count)
            return max(1, frame_num)
        return math.trunc(raw)

    def seek(self, target: TimecodeLike):
        self._check_open()
        if self._is_device:
            raise SeekError("Cannot seek if input is a device!")
        if not isinstance(target, FrameTimecode):
            target = FrameTimecode(target, self.frame_rate)
        if target < 0:
            raise ValueError("Target seek position cannot be negative!")
        target_secs = (self.base_timecode + target).seconds
        self._has_grabbed = False
        self._eof_timecode = None
        # A seek leaves the stream at an arbitrary timestamp, so subsequent reads must not
        # (re)capture the first-frame time origin. Seeking back to frame 0 restores sequential
        # reading; the origin captured on the original first frame still applies.
        self._sequential = target_secs <= 0.0
        # POS_FRAMES reports ticks rather than real frames after a seek on tick-rate containers.
        self._post_seek_tick_position = (
            target_secs > 0.0 and self._capture_rate != self._frame_rate
        )
        if target_secs > 0:
            # Tick-rate containers (e.g. remuxed AVI) measure POS_MSEC/POS_FRAMES at the raw
            # tick rate rather than the real frame rate. The wall-clock target time is still
            # expressed in seconds, so seek in time as usual but size the one-frame backstep and
            # the VFR correction window in *capture* ticks; frame numbers reported afterwards
            # are converted back to real frames in the `frame_number` property.
            one_frame_ms = 1000.0 / float(self._capture_rate)
            seek_ms = max(0.0, target_secs * 1000.0 - one_frame_ms)
            self._cap.set(cv2.CAP_PROP_POS_MSEC, seek_ms)
            self._has_grabbed = self._cap.grab()
            # The VFR correction assumes POS_MSEC shares the frame rate; on a tick-rate container
            # it advances at the capture (tick) rate, so the loop would spuriously grab extra
            # frames and overshoot the target. Frame numbers are converted back in `frame_number`.
            if self._has_grabbed and self._capture_rate == self._frame_rate:
                # VFR correction: set(CAP_PROP_POS_MSEC) converts time using avg_fps internally,
                # which can land ~1s too early for VFR video. Read forward until we reach the
                # intended position. The threshold (2x one_frame_ms) never triggers for CFR.
                actual_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                corrections = 0
                while actual_ms < seek_ms - 2.0 * one_frame_ms and corrections < 100:
                    if not self._cap.grab():
                        break
                    actual_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                    corrections += 1
            # If we seeked past the end, back up one frame.
            if not self._has_grabbed:
                pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
                # OpenCV 5 can report NaN here after a seek past EOF, which would crash
                # round()/trunc and hang the CLI decode thread (#380).
                if not math.isfinite(pos):
                    raise SeekError("Target frame is beyond end of video!")
                seek_pos = round(pos - 1.0)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, seek_pos))
                self._has_grabbed = self._cap.grab()
            # Record the seek anchor on a tick-rate container. The frame the next read() returns
            # is the 1-based frame (target.frame_num + 1) while the raw counter sits at a tick
            # value; storing both lets `frame_number` recover later frames as
            # target.frame_num + (raw_tick - anchor_tick), since every grab advances the raw
            # tick counter and the real frame number by exactly one.
            if self._has_grabbed and self._capture_rate != self._frame_rate:
                self._seek_anchor_ticks = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
                self._seek_anchor_frame = target.frame_num
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._seek_anchor_ticks = None
            self._seek_anchor_frame = None

    def reset(self):
        """Close and re-open the VideoStream (should be equivalent to calling `seek(0)`)."""
        self._check_open()
        self._cap.release()
        # Re-open without a frame-rate override when the rate was auto-detected, so a tick-rate
        # container is re-probed the same way it was on the initial open (e.g. an AVI whose
        # CAP_PROP_FPS reports 600 Hz). An explicit override is preserved as-is.
        self._open_capture(self._frame_rate if self._frame_rate_overridden else None)

    def close(self) -> None:
        """Close the video, releasing the underlying `cv2.VideoCapture`. Has no effect if the
        stream is already closed. See
        :meth:`VideoStream.close() <scenedetect.video_stream.VideoStream.close>` for details."""
        if self._closed:
            return
        # A released capture no longer reports the video's properties, so record them first.
        self._closed_frame_size = self.frame_size
        # Keep whatever frame count `duration` currently resolves to so it remains available
        # after the capture is released.
        if self._corrected_frame_count is not None:
            self._closed_frame_count = self._corrected_frame_count
        else:
            self._closed_frame_count = math.trunc(
                self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )
        self._closed_aspect_ratio = self.aspect_ratio
        self._closed = True
        self._cap.release()

    def read(self, decode: bool = True) -> np.ndarray | bool:
        self._check_open()
        if not self._cap.isOpened():
            return False
        # At the end of some VFR streams OpenCV reports an earlier CAP_PROP_POS_MSEC once grab()
        # fails, so keep `position` on the last frame read like the other backends do (#561).
        last_timecode = self.timecode if self._has_grabbed else None
        has_grabbed = self._cap.grab()
        # If we failed to grab the frame, retry a few times if required.
        if not has_grabbed:
            duration = self.duration
            if duration is not None and duration > 0 and self.position < (duration - 1):
                for _ in range(self._max_decode_attempts):
                    has_grabbed = self._cap.grab()
                    if has_grabbed:
                        break
            # Report previous failure in debug mode.
            if has_grabbed:
                self._decode_failures += 1
                logger.debug("Frame failed to decode.")
                if not self._warning_displayed and self._decode_failures > 1:
                    logger.warning("Failed to decode some frames, results may be inaccurate.")
        # We didn't manage to grab a frame even after retrying, so just return.
        if not has_grabbed:
            self._eof_timecode = last_timecode
            return False
        self._has_grabbed = True
        if self._sequential and self._time_origin_us is None:
            first_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            # Record the origin even when the first timestamp is still unknown (0.0): later
            # frames then keep their raw timestamp instead of subtracting a value captured at
            # an arbitrary position.
            self._time_origin_us = round(first_ms * 1000) if first_ms > 0.0 else 0
        self._eof_timecode = None
        # Need to make sure we actually grabbed a frame before calling retrieve.
        if decode and self._has_grabbed:
            _, frame = self._cap.retrieve()
            return frame
        return self._has_grabbed

    #
    # Private Methods
    #

    def _open_capture(self, frame_rate: FrameRate | None = None):
        """Opens capture referenced by this object and resets internal state."""
        if self._is_device:
            assert isinstance(self._path_or_device, int)
            if self._path_or_device < 0:
                raise ValueError("Invalid/negative device ID specified.")
            input_is_video_file = False
            capture_api = cv2.CAP_ANY
        else:
            assert isinstance(self._path_or_device, str)
            capture_api = _get_capture_api(self._path_or_device)
            input_is_video_file = not any(
                identifier in self._path_or_device
                for identifier in NON_VIDEO_FILE_INPUT_IDENTIFIERS
            )
            # We don't have a way of querying why opening a video fails (errors are logged at
            # least), so provide a better error message if we try to open a missing file.
            if input_is_video_file and not os.path.exists(self._path_or_device):
                raise OSError("Video file not found.")

        cap = cv2.VideoCapture(self._path_or_device, capture_api)
        if not cap.isOpened():
            raise VideoOpenFailure(
                "Ensure file is valid video and system dependencies are up to date.\n"
            )

        # Display an error if the video codec type seems unsupported (#86) as this indicates
        # potential video corruption, or may explain missing frames. We only perform this check
        # for video files on-disk (skipped for devices, image sequences, streams, etc...).
        codec_unsupported: bool = int(abs(cap.get(cv2.CAP_PROP_FOURCC))) == 0
        if codec_unsupported and input_is_video_file:
            logger.error(
                "Video codec detection failed. If output is incorrect:\n"
                "  - Re-encode the input video with ffmpeg\n"
                "  - Update OpenCV (pip install --upgrade opencv-python)\n"
                "  - Use the PyAV backend (--backend pyav)\n"
                "For details, see https://github.com/Breakthrough/PySceneDetect/issues/86"
            )

        # Ensure the framerate is correct to avoid potential divide by zero errors. This can be
        # addressed in the PyAV backend if required since it supports integer timebases.
        assert frame_rate is None or frame_rate > MAX_FPS_DELTA, (
            "Frame rate must be validated if set!"
        )
        reported_rate = cap.get(cv2.CAP_PROP_FPS)
        if frame_rate is None:
            frame_rate = reported_rate
            if frame_rate < MAX_FPS_DELTA:
                raise FrameRateUnavailable()

        self._cap: cv2.VideoCapture = cap
        self._frame_rate: Fraction = framerate_to_fraction(frame_rate)
        # The rate the capture itself applies when converting POS_MSEC to a frame position. This
        # equals the frame rate except for tick-rate containers (e.g. remuxed AVI), where the
        # capture uses the raw tick rate even though `frame_rate` reflects the real cadence.
        self._capture_rate: Fraction = framerate_to_fraction(reported_rate)
        self._has_grabbed = False
        self._eof_timecode: Timecode | None = None
        self._time_origin_us = None
        self._sequential = True
        self._post_seek_tick_position = False
        self._seek_anchor_ticks = None
        self._seek_anchor_frame = None
        self._corrected_frame_count = None
        # Remuxed AVI files expose a tick rate (e.g. 600 Hz) instead of the real frame rate.
        # The check is limited to AVI containers: other formats report a usable FPS directly,
        # and probing streams whose leading frames lack timestamps (e.g. MPEG-TS with missing
        # PTS, see issue #590) would derive a bogus rate from sparse timestamps. An explicit
        # frame-rate override always wins.
        if (
            not self._frame_rate_overridden
            and input_is_video_file
            and isinstance(self._path_or_device, str)
            and _is_avi_file(self._path_or_device)
            and reported_rate >= MAX_FPS_DELTA
        ):
            cadence = _probe_frame_cadence(cap)
            # Rewind whatever frames the probe grabbed.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if cadence is not None and cadence > MAX_FPS_DELTA:
                cadence_fraction = framerate_to_fraction(cadence)
                reported_fraction = framerate_to_fraction(reported_rate)
                # The reported rate only counts ticks when it differs substantially from the
                # rate the frame timestamps imply (same 2x rule as the PyAV backend).
                if float(reported_fraction) > 2.0 * float(cadence_fraction):
                    logger.debug(
                        "Reported frame rate %.3f fps is a tick rate; using %.3f fps derived"
                        " from frame timestamps.",
                        float(reported_fraction),
                        float(cadence_fraction),
                    )
                    self._frame_rate = cadence_fraction
                    self._correct_tick_frame_count(cadence_fraction, reported_fraction)
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)  # https://github.com/opencv/opencv/issues/26795

    def _correct_tick_frame_count(
        self, actual_rate: Fraction, reported_rate: Fraction
    ) -> None:
        """Reinterpret `CAP_PROP_FRAME_COUNT` when the capture counts time-base ticks.

        The counter still reflects the declared duration, so the shared non-exact rule converts
        that duration to frames at the real frame rate via :meth:`FrameTimecode.from_duration`,
        matching what the PyAV and MoviePy backends report for the same file.
        """
        tick_count = math.trunc(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if tick_count <= 0:
            return
        duration_secs = tick_count / float(reported_rate)
        self._corrected_frame_count = FrameTimecode.from_duration(
            duration_secs, actual_rate
        ).frame_num
        logger.debug(
            "Frame counter reports %d ticks at %.3f Hz (%.3fs); correcting to %d frames"
            " at %.3f fps.",
            tick_count,
            float(reported_rate),
            duration_secs,
            self._corrected_frame_count,
            float(actual_rate),
        )


class VideoCaptureAdapter(VideoStream):
    """Adapter for existing VideoCapture objects. Unlike VideoStreamCv2, this class supports
    VideoCaptures which may not support seeking.

    The VideoCapture remains owned by the caller: closing the adapter (e.g. by leaving a `with`
    block) does not release it.
    """

    def __init__(
        self,
        cap: cv2.VideoCapture,
        frame_rate: FrameRate | None = None,
        max_read_attempts: int = 5,
        framerate: float | None = None,
    ):
        """Create from an existing OpenCV VideoCapture object. Used for webcams, live streams,
        pipes, or other inputs which may not support seeking.

        Arguments:
            cap: The `cv2.VideoCapture` object to wrap. Must already be opened and ready to
                have `cap.read()` called on it.
            frame_rate: If set, overrides the detected frame rate. Takes precedence over
                `framerate`.
            max_read_attempts: Number of attempts to continue decoding the video
                after a frame fails to decode. This allows processing videos that
                have a few corrupted frames or metadata (in which case accuracy
                of detection algorithms may be lower). Once this limit is passed,
                decoding will stop and emit an error.
            framerate: [DEPRECATED] Use `frame_rate` instead. Retained as a deprecated
                alias for backwards compatibility; ignored when `frame_rate` is provided.

        Raises:
            ValueError: capture is not open, frame rate or max_read_attempts is invalid
        """
        super().__init__()

        if framerate is not None:
            warnings.warn(
                "`framerate` is deprecated and scheduled for removal in v0.9; "
                "use `frame_rate` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if frame_rate is None:
            frame_rate = framerate
        if frame_rate is not None and frame_rate < MAX_FPS_DELTA:
            raise ValueError(f"Specified frame rate ({float(frame_rate):f}) is invalid!")
        if max_read_attempts < 0:
            raise ValueError("Maximum decode attempts must be >= 0!")
        if not cap.isOpened():
            raise ValueError("Specified VideoCapture must already be opened!")
        if frame_rate is None:
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            if frame_rate < MAX_FPS_DELTA:
                raise FrameRateUnavailable()

        self._cap = cap
        self._frame_rate: Fraction = framerate_to_fraction(frame_rate)
        self._num_frames = 0
        self._max_read_attempts = max_read_attempts
        self._decode_failures = 0
        self._warning_displayed = False
        self._time_base: float = 0.0

    #
    # Backend-Specific Methods/Properties
    #

    @property
    def capture(self) -> cv2.VideoCapture:
        """Returns reference to underlying VideoCapture object. Use with caution.

        Prefer to use this property only to take ownership of the underlying cv2.VideoCapture object
        backing this object. Using the read/grab methods through this property are unsupported and
        will leave this object in an inconsistent state.
        """
        return self._cap

    #
    # VideoStream Methods/Properties
    #

    BACKEND_NAME = "opencv_adapter"
    """Unique name used to identify this backend."""

    @property
    def frame_rate(self) -> Fraction:
        """Framerate in frames/sec."""
        return self._frame_rate

    @property
    def path(self) -> str:
        """Always 'CAP_ADAPTER'."""
        return "CAP_ADAPTER"

    @property
    def name(self) -> str:
        """Always 'CAP_ADAPTER'."""
        return "CAP_ADAPTER"

    @property
    def is_seekable(self) -> bool:
        """Always False, as the underlying VideoCapture is assumed to not support seeking."""
        return False

    @property
    def frame_size(self) -> tuple[int, int]:
        """Reported size of each video frame in pixels as a tuple of (width, height)."""
        return (
            math.trunc(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            math.trunc(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def duration(self) -> FrameTimecode | None:
        """Duration of the stream as a FrameTimecode, or None if non terminating."""
        frame_count = math.trunc(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            return self.base_timecode + frame_count
        return None

    @property
    def aspect_ratio(self) -> float:
        """Display/pixel aspect ratio as a float (1.0 represents square pixels)."""
        return _get_aspect_ratio(self._cap)

    @property
    def position(self) -> FrameTimecode:
        if self.frame_number < 1:
            return self.base_timecode
        # Synthesize a Timecode from frame count and rational framerate.
        fps = self.frame_rate
        time_base = Fraction(1, fps.numerator)
        pts = (self.frame_number - 1) * fps.denominator
        timecode = Timecode(pts=pts, time_base=time_base)
        return FrameTimecode(timecode=timecode, fps=fps)

    @property
    def position_ms(self) -> float:
        if self._num_frames == 0:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_POS_MSEC) - self._time_base

    @property
    def frame_number(self) -> int:
        return self._num_frames

    def seek(self, target: TimecodeLike):
        """The underlying VideoCapture is assumed to not support seeking."""
        raise NotImplementedError("Seeking is not supported.")

    def reset(self):
        """Not supported."""
        raise NotImplementedError("Reset is not supported.")

    def read(self, decode: bool = True) -> np.ndarray | bool:
        if not self._cap.isOpened():
            return False
        has_grabbed = self._cap.grab()
        # If we failed to grab the frame, retry a few times if required.
        if not has_grabbed:
            for _ in range(self._max_read_attempts):
                has_grabbed = self._cap.grab()
                if has_grabbed:
                    break
            # Report previous failure in debug mode.
            if has_grabbed:
                self._decode_failures += 1
                logger.debug("Frame failed to decode.")
                if not self._warning_displayed and self._decode_failures > 1:
                    logger.warning("Failed to decode some frames, results may be inaccurate.")
        # We didn't manage to grab a frame even after retrying, so just return.
        if not has_grabbed:
            return False
        if self._num_frames == 0:
            self._time_base = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        self._num_frames += 1
        # Need to make sure we actually grabbed a frame before calling retrieve.
        if decode and self._num_frames > 0:
            _, frame = self._cap.retrieve()
            return frame
        return True
