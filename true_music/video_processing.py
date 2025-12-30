import json
import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from .context import get_config
from .pitch import detect_pitch_advanced


def _run_ffmpeg(args: List[str]) -> None:
    completed = subprocess.run(args, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"ffmpeg 执行失败: {stderr}")


def _run_ffprobe(args: List[str]) -> Dict:
    completed = subprocess.run(args, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"ffprobe 执行失败: {stderr}")
    return json.loads(completed.stdout or "{}")


def probe_video_info(video_path: str) -> Dict[str, float]:
    """获取视频分辨率、帧率、时长信息"""
    args = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        video_path,
    ]
    info = _run_ffprobe(args)
    stream = (info.get("streams") or [{}])[0]
    fmt = info.get("format") or {}
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    duration = float(fmt.get("duration") or 0.0)
    fps = _parse_fps(stream.get("r_frame_rate") or "0/0")
    return {"width": width, "height": height, "fps": fps, "duration": duration}


def _parse_fps(rate_text: str) -> float:
    if not rate_text or "/" not in rate_text:
        return 0.0
    num, den = rate_text.split("/", 1)
    try:
        num_f = float(num)
        den_f = float(den)
        return num_f / den_f if den_f != 0 else 0.0
    except ValueError:
        return 0.0


def extract_audio_to_wav(video_path: str, audio_path: str, sample_rate: int) -> None:
    args = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        audio_path,
    ]
    _run_ffmpeg(args)


def segment_audio(y: np.ndarray, sr: int) -> Tuple[List[Tuple[float, float]], Optional[str]]:
    """
    自动切割算法：
    1) 先做静音切分，得到有声区间
    2) 再在有声区间内用起始点检测做细分
    """
    config = get_config()
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    intervals = librosa.effects.split(
        y,
        top_db=config.silence_threshold_db,
        frame_length=2048,
        hop_length=512,
    )
    if len(intervals) == 0:
        return [], "未检测到有效有声区间"

    segments: List[Tuple[float, float]] = []
    for start, end in intervals:
        y_chunk = y[start:end]
        if len(y_chunk) == 0:
            continue
        onsets = librosa.onset.onset_detect(
            y=y_chunk,
            sr=sr,
            hop_length=512,
            backtrack=True,
            units="samples",
        )
        if len(onsets) == 0:
            segments.append((start / sr, end / sr))
            continue
        onset_samples = [start + int(o) for o in onsets]
        onset_samples = sorted(set(onset_samples))
        for i, onset in enumerate(onset_samples):
            seg_start = onset
            seg_end = onset_samples[i + 1] if i + 1 < len(onset_samples) else end
            duration = (seg_end - seg_start) / sr
            if duration >= config.min_clip_duration:
                segments.append((seg_start / sr, seg_end / sr))

    if not segments:
        return [], "起始点检测未得到有效片段"
    return segments, None


def cut_video_segment(
    video_path: str,
    output_path: str,
    start_sec: float,
    end_sec: float,
    width: int,
    height: int,
    fps: float,
) -> None:
    duration = max(0.0, end_sec - start_sec)
    vf = f"scale={width}:{height}"
    args = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        video_path,
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


def process_video_to_clips(video_path: str, auto_segment: bool = True) -> Dict[str, object]:
    """
    视频处理入口：
    - 提取音频
    - 自动切割
    - 逐段做音高识别，并返回片段信息
    """
    config = get_config()
    if not os.path.exists(video_path):
        return {"error": "视频文件不存在", "segments": []}

    video_info = probe_video_info(video_path)
    if video_info["width"] <= 0 or video_info["height"] <= 0:
        return {"error": "无法解析视频信息", "segments": []}

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    audio_path = os.path.join(config.video_dir, f"video_audio_{timestamp}.wav")
    extract_audio_to_wav(video_path, audio_path, config.sample_rate)

    y, sr = sf.read(audio_path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if auto_segment:
        segments, reason = segment_audio(y, sr)
    else:
        segments, reason = [(0.0, len(y) / sr)], "已关闭自动切割"
    if not segments:
        segments = [(0.0, len(y) / sr)]

    results = []
    for idx, (start_sec, end_sec) in enumerate(segments):
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        audio_clip = y[start_sample:end_sample]
        note_info = detect_pitch_advanced(audio_clip, sr)

        seg_filename = f"video_seg_{timestamp}_{idx:04d}.mp4"
        seg_path = os.path.join(config.video_clip_dir, seg_filename)
        cut_video_segment(
            video_path,
            seg_path,
            start_sec,
            end_sec,
            width=video_info["width"],
            height=video_info["height"],
            fps=video_info["fps"] or 30.0,
        )

        results.append(
            {
                "start": float(start_sec),
                "end": float(end_sec),
                "audio": audio_clip,
                "sample_rate": sr,
                "note_info": note_info,
                "video_path": seg_path,
                "video_info": video_info,
            }
        )

    return {
        "segments": results,
        "reason": reason,
        "audio_path": audio_path,
        "video_info": video_info,
    }
