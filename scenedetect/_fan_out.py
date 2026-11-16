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
"""Tee one VideoStream into N consumer streams sharing a single decode.

Used by the benchmark sweep harness to amortize video decoding across multiple detector
configurations running in parallel: one source decode feeds N consumer streams, each
read by an independent detection thread. The source is paced by the slowest consumer
(blocking ``put`` into bounded per-consumer queues), so peak memory is bounded by
``n * prefetch`` frames.

Shutdown and error propagation
------------------------------

Two distinct termination paths are supported:

* :meth:`FanOutVideoStream.abort` is a *clean cancellation*: the reader stops pulling
  new frames and every consumer drains the frames already buffered in its queue before
  receiving ordinary end-of-stream (``read()`` returns ``False``). No error is recorded.
* :meth:`FanOutVideoStream.fail` records a consumer-side failure and :meth:`abort` is
  likewise triggered when the source reader itself raises. In both cases every consumer
  first drains its buffered frames (nothing already admitted to a queue is dropped),
  then its next ``read()`` re-raises the recorded exception. Frames not yet read from
  the source at failure time are simply never decoded.

Frames already placed on a queue are never discarded: unblocking a reader stuck on a
full queue is done by stopping delivery to the *failed* consumer's queue only, never by
draining queues belonging to consumers that are still alive.

Internal API (underscore-prefixed module). Not part of the public surface.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from fractions import Fraction

import numpy as np

from scenedetect.common import FrameTimecode, TimecodeLike
from scenedetect.video_stream import SeekError, VideoStream

logger = logging.getLogger("pyscenedetect")

_EOF = object()
"""Sentinel placed on each consumer queue when the source reaches end-of-stream."""

# Bounded time for blocking queue operations. Every wait is retried while re-checking the
# stop flag so no thread can remain blocked past shutdown.
_QUEUE_TIMEOUT = 0.1


class FanOutVideoStream:
    """Drives one source :class:`VideoStream` and fans frames out to N consumer streams.

    Usage::

        source = open_video("video.mp4")
        fan = FanOutVideoStream(source, n=4)
        fan.start()
        try:
            for i in range(4):
                threading.Thread(target=worker, args=(fan.stream(i),)).start()
            # ... join workers ...
        finally:
            fan.close()

    The wrapper owns one background reader thread. Each ``stream(i)`` handle is a
    forward-only :class:`VideoStream` that reads from its own queue. ``seek``/``reset``
    on a consumer raise :class:`SeekError` -- to re-run a sweep over the same source,
    call ``source.reset()`` on the underlying stream and build a fresh
    :class:`FanOutVideoStream` for the next chunk.
    """

    def __init__(self, source: VideoStream, n: int, prefetch: int = 4):
        """
        Arguments:
            source: Already-opened ``VideoStream`` to read from.
            n: Number of consumer streams to expose. Must be >= 1.
            prefetch: Per-consumer queue depth. ``0`` is rendezvous (every frame waits
                for every consumer to take it); 4-8 absorbs jitter between consumers
                at the cost of up to ``n * prefetch`` resident frames.
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        if prefetch < 0:
            raise ValueError("prefetch must be >= 0")
        self._source = source
        # queue.Queue(maxsize=0) means unbounded, which would defeat back-pressure.
        # prefetch=0 therefore maps to a 1-deep buffer (shallow, not strict rendezvous).
        qsize = prefetch if prefetch > 0 else 1
        self._queues: list[queue.Queue] = [queue.Queue(maxsize=qsize) for _ in range(n)]
        self._consumers: list[_FanOutConsumer] = [_FanOutConsumer(self, i) for i in range(n)]
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        self._started = False
        self._closed = False
        # Error state is written under the lock; reads may be lock-free since assignments
        # are atomic under the GIL and the value only ever goes from None -> exception.
        self._state_lock = threading.Lock()
        self._reader_exc: BaseException | None = None
        self._consumer_exc: BaseException | None = None
        # Indices of consumers that have failed (or otherwise stopped reading). The reader
        # skips their queues, which is what unblocks a full-queue back-pressure stall
        # without discarding frames buffered for the consumers still alive.
        self._failed_consumers: set[int] = set()

    @property
    def num_consumers(self) -> int:
        """Number of consumer streams exposed by this wrapper."""
        return len(self._consumers)

    @property
    def error(self) -> BaseException | None:
        """The first exception that terminated the fan-out, or ``None`` while healthy.

        Set by :meth:`fail` (consumer-side failure) or by the reader thread when the
        source raises. Remains ``None`` after a plain :meth:`abort` or normal EOF.
        """
        if self._consumer_exc is not None:
            return self._consumer_exc
        return self._reader_exc

    def stream(self, i: int) -> VideoStream:
        """Return the i-th consumer ``VideoStream``."""
        return self._consumers[i]

    def start(self) -> None:
        """Spawn the reader thread. Idempotent; subsequent calls are no-ops."""
        if self._started:
            return
        self._started = True
        self._reader = threading.Thread(
            target=self._read_loop, name="FanOutVideoStream-reader", daemon=True
        )
        self._reader.start()

    def fail(self, exc: BaseException, consumer: int | None = None) -> None:
        """Record a consumer-side failure and initiate shutdown.

        The first exception wins; subsequent failures are logged but do not replace it.
        All consumers drain their already-buffered frames and then re-raise this
        exception from their next blocking ``read()``.

        Arguments:
            exc: The exception raised by the consumer.
            consumer: Index of the failing consumer, if known. Its queue stops receiving
                frames so the reader cannot stall on it. May be ``None`` for failures
                not tied to a specific consumer.
        """
        with self._state_lock:
            if self._consumer_exc is None and self._reader_exc is None:
                self._consumer_exc = exc
            else:
                logger.debug("Ignoring subsequent fan-out failure: %r", exc)
            if consumer is not None:
                self._failed_consumers.add(consumer)
        logger.debug("Fan-out consumer %s failed: %r", consumer, exc)
        # Signal the reader and release any queue it may currently be blocked on. The
        # queues of surviving consumers are left intact so their buffered frames survive.
        self.abort()

    def abort(self) -> None:
        """Request a clean stop without recording an error.

        The reader stops pulling new frames; each consumer drains the frames already
        buffered in its queue and then receives ordinary EOF (``read()`` returns
        ``False``). Idempotent and safe to call from any thread, including a consumer.
        """
        self._stop.set()

    def close(self) -> None:
        """Stop the reader thread and release resources. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self.abort()
        if self._reader is not None:
            self._reader.join(timeout=5.0)
            if self._reader.is_alive():
                # The reader only blocks on bounded-time queue operations and the stop
                # flag, so this indicates a broken source rather than an ordinary race.
                logger.error("Fan-out reader thread did not exit within 5 seconds.")

    # ------------------------------------------------------------------ #
    # Reader thread
    # ------------------------------------------------------------------ #

    def _read_loop(self) -> None:
        try:
            while not self._stop.is_set():
                frame = self._source.read()
                if frame is False:
                    break
                # Block per-consumer; the slowest consumer paces the source. Failed
                # consumers are skipped so one dead consumer cannot stall the shutdown.
                for i, q in enumerate(self._queues):
                    if i in self._failed_consumers:
                        continue
                    while not self._stop.is_set():
                        try:
                            q.put(frame, timeout=_QUEUE_TIMEOUT)
                            break
                        except queue.Full:
                            continue
                    if self._stop.is_set():
                        # Shutdown in progress (failure or abort). Stop while every frame
                        # already admitted to a queue remains there for its consumer.
                        return
        except BaseException as e:
            with self._state_lock:
                if self._reader_exc is None and self._consumer_exc is None:
                    self._reader_exc = e
                else:
                    logger.debug("Reader failed after shutdown was already requested: %r", e)
            logger.debug("Fan-out source reader raised: %r", e)
            self._stop.set()
        finally:
            self._send_sentinels()

    def _send_sentinels(self) -> None:
        """Deliver the terminal item to every consumer queue.

        Healthy consumers get the plain ``_EOF`` sentinel behind any frames still
        buffered (back-pressure is respected); failed consumers get it force-pushed so
        their (now-ownerless) queue cannot block reader shutdown.
        """
        for i, q in enumerate(self._queues):
            if i in self._failed_consumers:
                _force_put(q, _EOF)
                continue
            while True:
                try:
                    q.put(_EOF, timeout=_QUEUE_TIMEOUT)
                    break
                except queue.Full:
                    if self._stop.is_set():
                        # A surviving consumer stopped reading without calling fail()
                        # (e.g. its thread was killed). Drain only that consumer's queue
                        # so the reader can exit; no still-running consumer loses frames.
                        _force_put(q, _EOF)
                        break


def _force_put(q: queue.Queue, item: object) -> None:
    """Put ``item`` on ``q`` even when it is full, dropping one queued item if needed."""
    while True:
        try:
            q.put(item, timeout=_QUEUE_TIMEOUT)
            return
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                q.get_nowait()


class _FanOutConsumer(VideoStream):
    """One consumer-side handle exposed by :class:`FanOutVideoStream`.

    Forwards constant metadata (path, frame_rate, frame_size, etc.) to the source.
    Maintains its own ``frame_number`` / ``position`` -- both advance only when this
    consumer calls ``read()``, independent of the source's position or sibling
    consumers.
    """

    BACKEND_NAME = "fan_out"

    def __init__(self, parent: FanOutVideoStream, index: int):
        self._parent = parent
        self._index = index
        self._frame_number = 0
        self._eof = False

    @property
    def path(self) -> str:
        return self._parent._source.path

    @property
    def name(self) -> str:
        return self._parent._source.name

    @property
    def is_seekable(self) -> bool:
        return False

    @property
    def frame_rate(self) -> Fraction:
        return self._parent._source.frame_rate

    @property
    def duration(self) -> FrameTimecode | None:
        return self._parent._source.duration

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._parent._source.frame_size

    @property
    def aspect_ratio(self) -> float:
        return self._parent._source.aspect_ratio

    @property
    def decode_failures(self) -> int:
        return self._parent._source.decode_failures

    @property
    def frame_number(self) -> int:
        return self._frame_number

    @property
    def position(self) -> FrameTimecode:
        # Mirrors VideoStream contract: "frame 1 corresponds to presentation time 0;
        # returns 0 even if frame_number is 1."
        n = max(0, self._frame_number - 1)
        return FrameTimecode(timecode=n, fps=self.frame_rate)

    @property
    def position_ms(self) -> float:
        if self._frame_number == 0:
            return 0.0
        fps = self.frame_rate
        return float(1000 * (self._frame_number - 1) * fps.denominator) / float(fps.numerator)

    def read(self, decode: bool = True) -> np.ndarray | bool:
        if self._eof:
            # Mirror the VideoStream protocol: repeated reads after EOF keep returning
            # False; after an error keep raising the same exception.
            if self._parent.error is not None:
                raise self._parent.error
            return False
        item = self._parent._queues[self._index].get()
        if item is _EOF:
            self._eof = True
            error = self._parent.error
            if error is not None:
                raise error
            return False
        self._frame_number += 1
        # The source already decoded the frame; decode=False just suppresses returning it.
        if not decode:
            return True
        return item  # type: ignore[return-value]

    def reset(self) -> None:
        raise SeekError("FanOutVideoStream consumers are forward-only; reset the source instead.")

    def seek(self, target: TimecodeLike) -> None:
        del target
        raise SeekError("FanOutVideoStream consumers are forward-only; seeking is not supported.")
