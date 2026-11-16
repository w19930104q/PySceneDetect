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
"""PySceneDetect scenedetect.backend.opencv Tests

This file includes unit tests for the scenedetect.backend.opencv module that implements the
VideoStreamCv2 ('opencv') backend. These tests validate behaviour specific to this backend.

For VideoStream tests that validate conformance, see test_video_stream.py.
"""

import os

import cv2
import pytest

from scenedetect import ContentDetector, SceneManager
from scenedetect.backends.opencv import VideoCaptureAdapter, VideoStreamCv2

GROUND_TRUTH_CAPTURE_ADAPTER_TEST = [1, 90, 210]
GROUND_TRUTH_CAPTURE_ADAPTER_CALLBACK_TEST = [180, 394]


def test_open_image_sequence(test_image_sequence: str):
    """Test opening an image sequence. Currently, only VideoStreamCv2 supports this."""
    sequence = VideoStreamCv2(test_image_sequence, frame_rate=25.0)
    assert sequence.is_seekable
    assert sequence.frame_size[0] > 0 and sequence.frame_size[1] > 0
    assert sequence.duration is not None
    assert sequence.duration.frame_num == 30
    assert sequence.read() is not False
    sequence.seek(100)
    assert sequence.position == 29


@pytest.mark.skipif(os.name != "nt", reason="Workaround only applies on Windows.")
def test_open_png_sequence_skips_ffmpeg_probe(test_image_sequence: str, monkeypatch):
    """PNG inputs must be opened with `cv2.CAP_IMAGES` on Windows because certain versions of the
    opencv-python 5.x package cause intermittent segfaults on exit.

    See https://scenedetect.com/issue/575 for details."""
    calls = []
    real_capture = cv2.VideoCapture

    def spy(*args):
        calls.append(args)
        return real_capture(*args)

    monkeypatch.setattr(cv2, "VideoCapture", spy)
    sequence = VideoStreamCv2(test_image_sequence, frame_rate=25.0)
    sequence.reset()
    assert calls == [(test_image_sequence, cv2.CAP_IMAGES)] * 2
    assert sequence.capture.getBackendName() == "CV_IMAGES"
    assert sequence.read() is not False


def test_capture_adapter(test_movie_clip: str):
    """Test that the VideoCaptureAdapter works with SceneManager."""
    cap = cv2.VideoCapture(test_movie_clip)
    assert cap.isOpened()
    adapter = VideoCaptureAdapter(cap)
    assert adapter.read() is not False

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    assert scene_manager.detect_scenes(video=adapter, duration=adapter.base_timecode + 10.0)
    scenes = scene_manager.get_scene_list()
    assert len(scenes) == len(GROUND_TRUTH_CAPTURE_ADAPTER_TEST)
    assert [start.frame_num for (start, _) in scenes] == GROUND_TRUTH_CAPTURE_ADAPTER_TEST


def test_close_releases_capture(test_movie_clip: str):
    """`close()` must release the VideoCapture opened by the stream."""
    stream = VideoStreamCv2(test_movie_clip)
    capture = stream.capture
    assert capture.isOpened()
    stream.close()
    assert not capture.isOpened()


def test_capture_adapter_close_leaves_capture_open(test_movie_clip: str):
    """The VideoCapture wrapped by a VideoCaptureAdapter belongs to the caller, so closing the
    adapter (here by leaving a `with` block) must not release it."""
    cap = cv2.VideoCapture(test_movie_clip)
    try:
        with VideoCaptureAdapter(cap) as adapter:
            assert adapter.read() is not False
        assert cap.isOpened()
    finally:
        cap.release()


def test_capture_adapter_framerate_legacy_alias(test_movie_clip: str):
    """`framerate=` is the deprecated alias for `frame_rate=` on VideoCaptureAdapter."""
    cap = cv2.VideoCapture(test_movie_clip)
    assert cap.isOpened()
    with pytest.warns(DeprecationWarning, match="frame_rate"):
        legacy = VideoCaptureAdapter(cap, framerate=30.0)

    cap = cv2.VideoCapture(test_movie_clip)
    assert cap.isOpened()
    canonical = VideoCaptureAdapter(cap, frame_rate=30.0)
    assert canonical.frame_rate == legacy.frame_rate

    cap = cv2.VideoCapture(test_movie_clip)
    assert cap.isOpened()
    with pytest.warns(DeprecationWarning, match="frame_rate"):
        both = VideoCaptureAdapter(cap, frame_rate=30.0, framerate=24.0)
    assert both.frame_rate == canonical.frame_rate


def test_decode_failures_exposed(corrupt_video_file: str):
    """The private decode failure counters must be surfaced by the public property on both
    VideoStreamCv2 and VideoCaptureAdapter."""
    stream = VideoStreamCv2(corrupt_video_file)
    while stream.read(decode=False) is not False:
        pass
    assert stream.decode_failures == stream._decode_failures
    assert stream.decode_failures >= 0

    cap = cv2.VideoCapture(corrupt_video_file)
    assert cap.isOpened()
    adapter = VideoCaptureAdapter(cap)
    while adapter.read(decode=False) is not False:
        pass
    assert adapter.decode_failures == adapter._decode_failures
    assert adapter.decode_failures >= 0


def test_avi_tick_rate_corrected_to_real_frame_rate(test_no_pts_avi: str):
    """A remuxed AVI reports 600 Hz (a tick rate) through CAP_PROP_FPS even though the content
    runs at ~24 fps. The backend must derive the real cadence from frame timestamps."""
    raw = cv2.VideoCapture(test_no_pts_avi)
    assert raw.get(cv2.CAP_PROP_FPS) == 600.0
    raw.release()

    stream = VideoStreamCv2(test_no_pts_avi)
    try:
        assert 23.0 < float(stream.frame_rate) < 25.0
        # The tick counter (12012 at 600 Hz) is reinterpreted as the 480 real frames.
        assert stream.duration is not None
        assert stream.duration.frame_num == 480
    finally:
        stream.close()


def test_avi_first_frame_anchored_at_zero(test_no_pts_avi: str):
    """DTS-only AVI packets report a nonzero timestamp for the first frame via POS_MSEC; the
    backend subtracts it so the first decoded frame sits at time 0 like the other backends."""
    stream = VideoStreamCv2(test_no_pts_avi)
    try:
        assert stream.read(decode=False) is not False
        assert stream.position == stream.base_timecode
        assert stream.position_ms == pytest.approx(0.0, abs=0.1)
        assert stream.frame_number == 1
        # The origin is captured only while reading sequentially from the start.
        assert stream._time_origin_us is not None and stream._time_origin_us > 0
    finally:
        stream.close()


def test_avi_seek_does_not_recapture_time_origin(test_no_pts_avi: str):
    """A seek lands on an arbitrary timestamp and must not replace the first-frame origin,
    otherwise every later position would be shifted by the seek point."""
    stream = VideoStreamCv2(test_no_pts_avi)
    try:
        assert stream.read(decode=False) is not False
        origin = stream._time_origin_us
        stream.seek(50)
        assert stream._sequential is False
        assert stream.read(decode=False) is not False
        # The origin is unchanged and the position reflects the seek, not a re-anchored zero.
        assert stream._time_origin_us == origin
        assert stream.frame_number == 51
        # Seeking back to the start restores sequential reading.
        stream.seek(0)
        assert stream._sequential is True
    finally:
        stream.close()


def test_normal_mp4_frame_rate_untouched(test_movie_clip: str):
    """The tick-rate probe must not alter a normal MP4 whose CAP_PROP_FPS is already correct."""
    stream = VideoStreamCv2(test_movie_clip)
    try:
        assert float(stream.frame_rate) == pytest.approx(23.976, abs=0.001)
        assert stream._corrected_frame_count is None
        assert stream.duration is not None
        assert stream.duration.frame_num == 1980
    finally:
        stream.close()


def test_explicit_frame_rate_disables_tick_probe(test_no_pts_avi: str):
    """An explicit frame-rate override always wins and suppresses the automatic probe."""
    stream = VideoStreamCv2(test_no_pts_avi, frame_rate=30.0)
    try:
        assert float(stream.frame_rate) == pytest.approx(30.0)
        assert stream._corrected_frame_count is None
    finally:
        stream.close()
