import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple

from .context import get_config


def _run_ffmpeg(args: List[str]) -> None:
    completed = subprocess.run(args, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"ffmpeg 执行失败: {stderr}")


def _ensure_segment_duration(
    input_path: str,
    output_path: str,
    duration: float,
    width: int,
    height: int,
    fps: float,
) -> None:
    vf = f"scale={width}:{height},tpad=stop_mode=clone:stop_duration={duration:.3f}"
    args = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-t",
        f"{duration:.3f}",
        "-vf",
        vf,
        "-r",
        f"{fps:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-an",
        output_path,
    ]
    _run_ffmpeg(args)


def _make_black_segment(
    output_path: str,
    duration: float,
    width: int,
    height: int,
    fps: float,
) -> None:
    args = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={width}x{height}:r={fps:.3f}:d={duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-an",
        output_path,
    ]
    _run_ffmpeg(args)


def compose_video_timeline(
    timeline: List[Tuple[float, float, Optional[Dict]]],
    default_video_info: Dict,
    audio_path: Optional[str] = None,
) -> str:
    """
    timeline: [(start_time_seconds, duration_seconds, clip_metadata_or_none), ...]
    """
    config = get_config()
    if not timeline:
        raise RuntimeError("没有可用的视频片段用于合成")

    width = int(default_video_info.get("width") or 1280)
    height = int(default_video_info.get("height") or 720)
    fps = float(default_video_info.get("fps") or 30.0)

    temp_dir = os.path.join(config.output_dir, "video_tmp")
    os.makedirs(temp_dir, exist_ok=True)

    segment_files = []
    current_time = 0.0
    for idx, (start_time, duration, clip_meta) in enumerate(timeline):
        if duration <= 0:
            continue

        if start_time > current_time:
            gap_duration = start_time - current_time
            gap_path = os.path.join(temp_dir, f"gap_{idx:04d}.mp4")
            _make_black_segment(gap_path, gap_duration, width, height, fps)
            segment_files.append(gap_path)
            current_time = start_time

        seg_path = os.path.join(temp_dir, f"seg_{idx:04d}.mp4")
        if clip_meta and clip_meta.get("video_path") and os.path.exists(
            clip_meta["video_path"]
        ):
            _ensure_segment_duration(
                clip_meta["video_path"], seg_path, duration, width, height, fps
            )
        else:
            _make_black_segment(seg_path, duration, width, height, fps)
        segment_files.append(seg_path)
        current_time += duration

    list_path = os.path.join(temp_dir, "concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for path in segment_files:
            f.write(f"file '{path}'\n")

    output_name = f"composition_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    output_path = os.path.join(config.output_dir, output_name)
    video_only_path = os.path.join(temp_dir, f"video_only_{output_name}")
    args = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c",
        "copy",
        video_only_path,
    ]
    _run_ffmpeg(args)

    if audio_path and os.path.exists(audio_path):
        mux_args = [
            "ffmpeg",
            "-y",
            "-i",
            video_only_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            output_path,
        ]
        _run_ffmpeg(mux_args)
        return output_path

    return video_only_path
