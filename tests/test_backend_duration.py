#
#            PySceneDetect: Python-Based Video Scene Detector
#   -------------------------------------------------------------------
#     [  Site:    https://scenedetect.com                           ]
#     [  Docs:    https://scenedetect.com/docs/                     ]
#     [  Github:  https://github.com/Breakthrough/PySceneDetect/    ]
#
# Copyright (C) 2026 Brandon Castellano <http://www.bcastell.com>.
# PySceneDetect is licensed under the BSD 3-Clause License; see the
# included LICENSE file, or visit one of the above pages for details.
#
"""Cross-backend duration/timecode consistency tests.

The OpenCV, PyAV, and MoviePy backends must report the same duration and frame-based timecodes
for a given input, otherwise sharding a video by frame range produces different cuts depending
on the backend. Two fixtures are deliberately awkward:

* ``issue-590-bframes.avi`` - an ffmpeg remux whose stream counter stores time-base *ticks*
  (12012 ticks at 600 Hz) rather than the real 480 frames, and whose packets only carry decode
  timestamps (no PTS).
* ``issue-590-missing-pts.ts`` - an MPEG-TS whose leading and some later packets have their
  PTS/DTS stripped, so no exact frame count is available and every backend must degrade to the
  same ``duration_seconds * frame_rate`` estimate.

These tests pin the shared rules in :mod:`scenedetect.common` (see :func:`duration_in_frames`
and :meth:`FrameTimecode.from_duration`) rather than any one backend's container quirks.
"""

import typing as ty

import pytest

from scenedetect import FrameTimecode
from scenedetect.backends import VideoStreamAv, VideoStreamMoviePy
from scenedetect.backends.opencv import VideoStreamCv2
from scenedetect.video_stream import VideoStream

# Backends present in this environment. MoviePy is an optional dependency.
BACKENDS: list[type[VideoStream]] = [VideoStreamCv2, VideoStreamAv]
if VideoStreamMoviePy is not None:
    BACKENDS.append(VideoStreamMoviePy)

# Filter for warnings we ignore from VideoStreamMoviePy (warnings come from FFMPEG_VideoReader).
MOVIEPY_WARNING_FILTER = "ignore:.*Using the last valid frame instead.:UserWarning"

pytestmark = pytest.mark.filterwarnings(MOVIEPY_WARNING_FILTER)


def _read_all(video: VideoStream) -> tuple[int, FrameTimecode, FrameTimecode]:
    """Return (frame count, first position, last position) by decoding the whole stream."""
    first = video.base_timecode
    last = video.base_timecode
    count = 0
    while video.read(decode=False) is not False:
        count += 1
        if count == 1:
            first = video.position
        last = video.position
    return count, first, last


#
# Remuxed AVI with a tick-rate frame counter and DTS-only packets.
#


def test_avi_frame_rate_matches_real_cadence(test_no_pts_avi: str, auto_close):
    """No backend may report the 600 Hz tick rate as the video frame rate. The content was
    encoded at 24000/1001 (~23.976 fps); OpenCV derives 24/1 from frame timestamps because the
    AVI header only stores integer microsecond spacing, while PyAV/MoviePy recover the codec
    rate. Both describe the same ~24 fps cadence within one frame period."""
    rates = {}
    for backend in BACKENDS:
        video = auto_close(backend(test_no_pts_avi))
        rates[backend.BACKEND_NAME] = video.frame_rate
        assert 23.0 < float(video.frame_rate) < 25.0
    # The OpenCV integer-spaced rate and the rational codec rate differ by less than 0.1%.
    assert abs(float(rates["opencv"]) - float(rates["pyav"])) < 0.05


def test_avi_duration_is_real_frame_count(test_no_pts_avi: str, auto_close):
    """The 12012-tick counter must be reinterpreted: every backend reports a duration of 480
    frames (MoviePy may be off by a single frame because ffmpeg's reported duration in seconds
    is rounded at the container boundary)."""
    durations = {}
    for backend in BACKENDS:
        video = auto_close(backend(test_no_pts_avi))
        duration = video.duration
        assert isinstance(duration, FrameTimecode)
        durations[backend.BACKEND_NAME] = duration.frame_num
        assert duration.frame_num == pytest.approx(480, abs=1)
    # OpenCV and PyAV must agree exactly, since both derive 480 from the 20 s duration.
    assert durations["opencv"] == durations["pyav"] == 480


def test_avi_first_and_last_timecodes_aligned(test_no_pts_avi: str, auto_close):
    """The first frame must sit at time 0 on every backend (DTS-only AVI otherwise reports a
    nonzero first timestamp via OpenCV), and the last frame must be near 20 s."""
    for backend in BACKENDS:
        video = auto_close(backend(test_no_pts_avi))
        count, first, last = _read_all(video)
        # MoviePy's ffmpeg pipe signals EOF one frame early on this clip (#461).
        assert count == pytest.approx(480, abs=1)
        assert first == video.base_timecode
        # Last frame presentation time ~= (count - 1) frame periods, just under 20 s.
        assert last.seconds == pytest.approx((count - 1) / float(video.frame_rate), abs=0.05)


def test_avi_duration_available_after_close(test_no_pts_avi: str, auto_close):
    """Duration must remain available, and unchanged, after the stream is closed."""
    for backend in BACKENDS:
        video = auto_close(backend(test_no_pts_avi))
        before = video.duration
        video.close()
        after = video.duration
        assert isinstance(after, FrameTimecode)
        assert after.frame_num == before.frame_num


#
# MPEG-TS with missing PTS/DTS: no exact frame count, shared seconds-based fallback.
#


def test_ts_frame_rate_consistent(test_missing_pts_ts: str, auto_close):
    """All backends detect the same nominal frame rate for the TS."""
    for backend in BACKENDS:
        video = auto_close(backend(test_missing_pts_ts))
        assert float(video.frame_rate) == pytest.approx(23.976, abs=0.001)


def test_ts_duration_uses_shared_fallback(test_missing_pts_ts: str, auto_close):
    """OpenCV and PyAV must report the identical duration: both only know the ~9.55 s
    container duration and convert it with the shared round-to-nearest-frame rule."""
    durations = {}
    for backend in BACKENDS:
        video = auto_close(backend(test_missing_pts_ts))
        duration = video.duration
        assert isinstance(duration, FrameTimecode)
        durations[backend.BACKEND_NAME] = duration
    assert durations["opencv"] == durations["pyav"]
    assert durations["opencv"].frame_num == 229
    # MoviePy's ffmpeg pipe over-estimates streams whose PTS was stripped (it re-times
    # duplicated packets), so it is only required to stay within a small frame margin rather
    # than match exactly.
    if VideoStreamMoviePy is not None:
        assert durations["moviepy"].frame_num == pytest.approx(229, abs=8)


def test_ts_first_timecode_is_zero(test_missing_pts_ts: str, auto_close):
    """Despite the stripped leading timestamps, the first decoded frame is anchored at 0."""
    for backend in BACKENDS:
        video = auto_close(backend(test_missing_pts_ts))
        assert video.read(decode=False) is not False
        assert video.position == video.base_timecode
        assert video.position_ms == pytest.approx(0.0, abs=0.1)
        assert video.frame_number == 1


#
# Shared conversion-rule pinning against backend-independent values.
#


def test_duration_fallback_rule_is_deterministic():
    """The same (seconds, fps) inputs must always yield the same frame count, and the result is
    a frame-accurate FrameTimecode regardless of which backend would use it."""
    fps = 24000 / 1001
    expected = round(9.551208 * fps)
    tc = FrameTimecode.from_duration(9.551208, fps)
    assert tc.frame_num == expected == 229
    # A second construction yields an equal timecode.
    assert FrameTimecode.from_duration(9.551208, fps) == tc
