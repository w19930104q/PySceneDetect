#
#            PySceneDetect: Python-Based Video Scene Detector
#   -------------------------------------------------------------------
#     [  Site:    https://scenedetect.com                           ]
#     [  Docs:    https://scenedetect.com/docs/                     ]
#     [  Github:  https://github.com/Breakthrough/PySceneDetect/    ]
#
# Copyright (C) 2025 Brandon Castellano <http://www.bcastell.com>.
# PySceneDetect is licensed under the BSD 3-Clause License; see the
# included LICENSE file, or visit one of the above pages for details.
#
# This software may also invoke mkvmerge or FFmpeg, if available.
# FFmpeg is a trademark of Fabrice Bellard.
# mkvmerge is Copyright (C) 2005-2016, Matroska.
# Certain distributions of PySceneDetect may include the above software;
# see the included LICENSE-FFMPEG and LICENSE-MKVMERGE files.
#
"""The ``scenedetect.output.video`` module contains functions to split existing videos into clips
using ffmpeg or mkvmerge.

These programs can be obtained from following URLs (note that mkvmerge is a part mkvtoolnix):

 * FFmpeg:   [ https://ffmpeg.org/download.html ]
 * mkvmerge: [ https://mkvtoolnix.download/downloads.html ]

If you are a Linux user, you can likely obtain the above programs from your package manager.

Once installed, ensure the program can be accessed system-wide by calling the `mkvmerge` or `ffmpeg`
command from a terminal/command prompt. PySceneDetect will automatically use whichever program is
available on the computer, depending on the specified command-line options.
"""

import logging
import re
import time
import typing as ty
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from scenedetect.common import FrameTimecode, TimecodePair
from scenedetect.platform import (
    CommandTooLong,
    Template,
    get_ffmpeg_path,
    get_mkvmerge_path,
    invoke_command,
    tqdm,
)

logger = logging.getLogger("pyscenedetect")

_COMMAND_TOO_LONG_STRING = """
Cannot split video due to too many scenes (resulting command
is too large to process). To work around this issue, you can
split the video manually by exporting a list of cuts with the
`list-scenes` command.
See https://github.com/Breakthrough/PySceneDetect/issues/164
for details.  Sorry about that!
"""

# TODO: Resolve this on first use (e.g., functools.cache on the getter) rather than at import
# time, so that importing this module doesn't spawn an ffmpeg subprocess.
_FFMPEG_PATH: str | None = get_ffmpeg_path()
"""Relative path to the ffmpeg binary on this system, if any (will be None if not available)."""

_DEFAULT_FFMPEG_ARGS = (
    "-map 0:v:0 -map 0:a? -map 0:s? -c:v libx264 -preset veryfast -crf 22 -c:a aac"
)
"""Default arguments passed to ffmpeg when invoking the `split_video_ffmpeg` function."""

##
## Command Availability Checking Functions
##


def is_mkvmerge_available() -> bool:
    """Is mkvmerge Available: Gracefully checks if mkvmerge command is available.

    Returns:
        True if `mkvmerge` can be invoked, False otherwise.
    """
    return get_mkvmerge_path() is not None


def is_ffmpeg_available() -> bool:
    """Is ffmpeg Available: Gracefully checks if ffmpeg command is available.

    Returns:
        True if `ffmpeg` can be invoked, False otherwise.
    """
    return _FFMPEG_PATH is not None


##
## Output Naming
##


@dataclass
class VideoMetadata:
    """Information about the video being split."""

    name: str
    """Expected name of the video. May differ from `path`."""
    path: Path
    """Path to the input file."""
    total_scenes: int
    """Total number of scenes that will be written."""


@dataclass
class SceneMetadata:
    """Information about the scene being extracted."""

    index: int
    """0-based index of this scene."""
    start: FrameTimecode
    """First frame."""
    end: FrameTimecode
    """Last frame."""


PathFormatter = ty.Callable[[VideoMetadata, SceneMetadata], str]

# Minimum number of digits used when formatting scene numbers (e.g. scene 1 -> "001").
MIN_SCENE_NUMBER_DIGITS = 3

# Template variables that describe a single scene rather than the whole video. mkvmerge cannot
# expand these because it appends its own scene-number suffix to one shared output path.
_SCENE_VARIABLES = (
    "SCENE_NUMBER",
    "START_TIME",
    "END_TIME",
    "START_FRAME",
    "END_FRAME",
    "START_PTS",
    "END_PTS",
)


def _strip_scene_variables(template: str) -> str:
    """Remove per-scene `$VARIABLE` placeholders from an output filename template.

    mkvmerge derives every output name from one shared base path plus its own scene-number
    suffix, so per-scene variables cannot differ between its outputs. A separator directly
    before or after a removed variable is removed as well to avoid dangling or doubled
    separators (e.g. ``"$VIDEO_NAME-Scene-$SCENE_NUMBER"`` -> ``"$VIDEO_NAME-Scene"`` and
    ``"$VIDEO_NAME-$START_PTS-$END_PTS"`` -> ``"$VIDEO_NAME"``).
    """
    dot_pos = template.rfind(".")
    stem = template if dot_pos < 0 else template[:dot_pos]
    extension = "" if dot_pos < 0 else template[dot_pos:]
    for variable in _SCENE_VARIABLES:
        # Match both "$VAR" and "${VAR}". A separator between two variables is consumed with
        # the first one, so chains collapse without dangling or doubled separators.
        placeholder = rf"(?:{variable}|\{{{variable}\}})"
        stem = re.sub(rf"[-_]\${placeholder}(?=[-_]|$)", "", stem)
        stem = re.sub(rf"\${placeholder}[-_]?", "", stem)
    return stem + extension


def scene_number_width(total_scenes: int) -> int:
    """Number of digits required to represent every scene number, never below
    :data:`MIN_SCENE_NUMBER_DIGITS`."""
    # Count digits directly instead of using `floor(log10(...)) + 1`: the latter loses a digit at
    # exact powers of ten (e.g. 1000) due to floating-point rounding of log10().
    width = len(str(max(total_scenes, 1)))
    return max(MIN_SCENE_NUMBER_DIGITS, width)


def format_scene_number(total_scenes: int, index: int) -> str:
    """Format the 0-based scene `index` as a zero-padded, 1-based scene number whose width is
    shared by every scene in the output (e.g. ``"001"`` for 3 scenes, ``"0001"`` for 1000)."""
    return ("%0" + str(scene_number_width(total_scenes)) + "d") % (index + 1)


def timecode_to_ms(timecode: FrameTimecode) -> int:
    """Convert a timecode to a whole number of milliseconds.

    Uses the exact presentation time (``pts * time_base``), which is precise for VFR video,
    instead of deriving the value back from floating-point seconds. For CFR timecodes this is
    equivalent to rounding the frame time (``frame_num / frame_rate``) since ``pts`` is the frame
    number and ``time_base`` equals ``1 / frame_rate`` in that case.
    """
    return round(timecode.pts * timecode.time_base * 1000)


def default_formatter(template: str) -> PathFormatter:
    """Formats filenames using a template string which allows the following variables:

    `$VIDEO_NAME`, `$SCENE_NUMBER`, `$START_TIME`, `$END_TIME`, `$START_FRAME`, `$END_FRAME`,
    `$START_PTS`, `$END_PTS` (presentation timestamp in milliseconds, accurate for VFR video)
    """

    def formatter(video: VideoMetadata, scene: SceneMetadata) -> str:
        return Template(template).safe_substitute(
            VIDEO_NAME=video.name,
            SCENE_NUMBER=format_scene_number(video.total_scenes, scene.index),
            START_TIME=str(scene.start.get_timecode().replace(":", ";")),
            END_TIME=str(scene.end.get_timecode().replace(":", ";")),
            START_FRAME=str(scene.start.frame_num),
            END_FRAME=str(scene.end.frame_num),
            START_PTS=str(timecode_to_ms(scene.start)),
            END_PTS=str(timecode_to_ms(scene.end)),
        )

    return formatter


##
## Split Video Functions
##


def split_video_mkvmerge(
    input_video_path: str,
    scene_list: Sequence[TimecodePair],
    output_dir: str | Path | None = None,
    output_file_template: str = "$VIDEO_NAME.mkv",
    video_name: str | None = None,
    show_output: bool = False,
    suppress_output=None,
) -> int:
    """Split `input_video_path` using `mkvmerge` based on the scenes in `scene_list`.

    Arguments:
        input_video_path: Path to the video to be split.
        scene_list : List of scenes as pairs of FrameTimecodes denoting the start/end times.
        output_dir: Directory to output videos. If not set, output will be in working directory.
        output_file_template: Template to use for generating output files. Only the `$VIDEO_NAME`
            variable can differ per output: mkvmerge writes all parts from one shared output path
            and appends the scene number prefixed with a dash (``"-001"``, ``"-002"``, ...) to
            each file itself. Per-scene variables (`$SCENE_NUMBER`, `$START_TIME`, `$END_TIME`,
            `$START_FRAME`, `$END_FRAME`, `$START_PTS`, `$END_PTS`) cannot be represented in
            mkvmerge output names and are stripped (along with any adjacent separators) before
            the path is passed to mkvmerge. Use :func:`split_video_ffmpeg` for per-scene names.
        video_name: Name of the video to be substituted in output_file_template for
            $VIDEO_NAME. If not specified, will be obtained from the filename.
        show_output: If False, adds the --quiet flag when invoking `mkvmerge`.
        suppress_output: [DEPRECATED] DO NOT USE. For backwards compatibility only.
    Returns:
        Return code of invoking mkvmerge (0 on success). If scene_list is empty, will
        still return 0, but no commands will be invoked.
    """
    # Handle backwards compatibility with v0.5 API.
    if isinstance(input_video_path, list):
        logger.error("Using a list of paths is deprecated. Pass a single path instead.")
        if len(input_video_path) > 1:
            raise ValueError("Concatenating multiple input videos is not supported.")
        input_video_path = input_video_path[0]
    if suppress_output is not None:
        logger.error("suppress_output is deprecated, use show_output instead.")
        show_output = not suppress_output

    if not scene_list:
        return 0

    if video_name is None:
        video_name = Path(input_video_path).stem

    # mkvmerge takes a single output path and appends "-001", "-002", ... to every resulting
    # file itself; it cannot name outputs differently per scene. Per-scene template variables
    # ($SCENE_NUMBER, $START_PTS, ...) therefore cannot be represented in mkvmerge output names
    # and are stripped from the template together with any separator that would be left dangling.
    # mkvmerge's own suffixes are zero-padded to 3 digits, which matches the ffmpeg path's minimum
    # width for up to 999 scenes; the single-scene suffix below uses the shared formatter width.
    template = _strip_scene_variables(output_file_template)
    output_path = Path(Template(template).safe_substitute(VIDEO_NAME=video_name))
    if output_dir:
        output_path = Path(output_dir) / output_path
    logger.info(f"Splitting video with mkvmerge, path template: {output_path}")
    # If there is only one scene, mkvmerge omits the scene-number suffix. Append it ourselves
    # using the same width the formatter would have used so the name matches the multi-scene
    # output (and the ffmpeg path) exactly.
    if len(scene_list) == 1:
        output_path = output_path.with_stem(
            output_path.stem + "-" + format_scene_number(len(scene_list), 0)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    call_list = ["mkvmerge"]
    if not show_output:
        call_list.append("--quiet")
    call_list += [
        "-o",
        str(output_path),
        "--split",
        "parts:{}".format(
            ",".join(
                [
                    f"{start_time.get_timecode()}-{end_time.get_timecode()}"
                    for start_time, end_time in scene_list
                ]
            )
        ),
        input_video_path,
    ]
    total_frames = scene_list[-1][1].frame_num - scene_list[0][0].frame_num
    processing_start_time = time.time()
    ret_val = 0
    try:
        # TODO: Capture stdout/stderr and show that if the command fails.
        ret_val = invoke_command(call_list)
        if show_output:
            logger.info(
                "Average processing speed %.2f frames/sec.",
                float(total_frames) / (time.time() - processing_start_time),
            )
    except CommandTooLong:
        logger.error(_COMMAND_TOO_LONG_STRING)
    except OSError:
        logger.error(
            "mkvmerge could not be found on the system."
            " Please install mkvmerge to enable video output support."
        )
    if ret_val != 0:
        logger.error("Error splitting video (mkvmerge returned %d).", ret_val)
    return ret_val


def split_video_ffmpeg(
    input_video_path: str,
    scene_list: Sequence[TimecodePair],
    output_dir: str | Path | None = None,
    output_file_template: str = "$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4",
    video_name: str | None = None,
    arg_override: str = _DEFAULT_FFMPEG_ARGS,
    show_progress: bool = False,
    show_output: bool = False,
    suppress_output=None,
    hide_progress=None,
    formatter: PathFormatter | None = None,
) -> int:
    """Split `input_video_path` using `ffmpeg` based on the scenes in `scene_list`.

    Arguments:
        input_video_path: Path to the video to be split.
        scene_list: List of scenes (pairs of FrameTimecodes) denoting the start/end of each scene.
        output_dir: Directory to output videos. If not set, output will be in working directory.
        output_file_template: Template to use for generating output filenames.
            The following variables will be replaced in the template for each scene:
            $VIDEO_NAME, $SCENE_NUMBER, $START_TIME, $END_TIME, $START_FRAME, $END_FRAME
        video_name: Name of the video to be substituted in output_file_template. If not
            passed will be calculated from input_video_path automatically.
        arg_override: Allows overriding the arguments passed to ffmpeg for encoding.
        show_progress: If True, will show progress bar provided by tqdm (if installed).
        show_output: If True, will show output from ffmpeg for first split.
        suppress_output: [DEPRECATED] DO NOT USE. For backwards compatibility only.
        hide_progress: [DEPRECATED] DO NOT USE. For backwards compatibility only.
        formatter: Custom formatter callback. Overrides `output_file_template`.

    Returns:
        Return code of invoking ffmpeg (0 on success). If scene_list is empty, will
        still return 0, but no commands will be invoked.
    """
    # Handle backwards compatibility with v0.5 API.
    if isinstance(input_video_path, list):
        logger.error("Using a list of paths is deprecated. Pass a single path instead.")
        if len(input_video_path) > 1:
            raise ValueError("Concatenating multiple input videos is not supported.")
        input_video_path = input_video_path[0]
    if suppress_output is not None:
        logger.error("suppress_output is deprecated, use show_output instead.")
        show_output = not suppress_output
    if hide_progress is not None:
        logger.error("hide_progress is deprecated, use show_progress instead.")
        show_progress = not hide_progress

    if not scene_list:
        return 0

    logger.info("Splitting video with ffmpeg, output path template:\n  %s", output_file_template)
    if output_dir:
        logger.info("Output folder:\n  %s", output_dir)

    if video_name is None:
        video_name = Path(input_video_path).stem

    arg_override = arg_override.replace('\\"', '"')

    ret_val = 0
    ffmpeg_args = arg_override.split(" ")

    if formatter is None:
        formatter = default_formatter(output_file_template)
    video_metadata = VideoMetadata(
        name=video_name, path=Path(input_video_path), total_scenes=len(scene_list)
    )

    try:
        progress_bar = None
        total_frames = scene_list[-1][1].frame_num - scene_list[0][0].frame_num
        if show_progress:
            progress_bar = tqdm(total=total_frames, unit="frame", miniters=1, dynamic_ncols=True)
        processing_start_time = time.time()
        for i, (start_time, end_time) in enumerate(scene_list):
            duration = end_time - start_time
            scene_metadata = SceneMetadata(index=i, start=start_time, end=end_time)
            output_path = Path(formatter(video_metadata, scene_metadata))
            if output_dir:
                output_path = Path(output_dir) / output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Gracefully handle case where FFMPEG_PATH might be unset.
            call_list = [_FFMPEG_PATH if _FFMPEG_PATH is not None else "ffmpeg"]
            if not show_output:
                call_list += ["-v", "quiet"]
            elif i > 0:
                # Only show ffmpeg output for the first call, which will display any
                # errors if it fails, and then break the loop. We only show error messages
                # for the remaining calls.
                call_list += ["-v", "error"]
            call_list += [
                "-nostdin",
                "-y",
                "-ss",
                str(start_time.seconds),
                "-i",
                input_video_path,
                "-t",
                str(duration.seconds),
            ]
            call_list += ffmpeg_args
            call_list += ["-sn"]
            call_list += [str(output_path)]
            ret_val = invoke_command(call_list)
            if show_output and i == 0 and len(scene_list) > 1:
                logger.info(
                    "Output from ffmpeg for Scene 1 shown above, splitting remaining scenes..."
                )
            if ret_val != 0:
                # TODO: Capture stdout/stderr and display it on any failed calls.
                logger.error("Error splitting video (ffmpeg returned %d).", ret_val)
                break
            if progress_bar:
                progress_bar.update(duration.frame_num)

        if progress_bar:
            progress_bar.close()
        if show_output:
            logger.info(
                "Average processing speed %.2f frames/sec.",
                float(total_frames) / (time.time() - processing_start_time),
            )

    except CommandTooLong:
        logger.error(_COMMAND_TOO_LONG_STRING)
    except OSError:
        logger.error(
            "ffmpeg could not be found on the system."
            " Please install ffmpeg to enable video output support."
        )
    return ret_val
