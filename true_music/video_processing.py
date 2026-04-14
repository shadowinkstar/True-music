import json
import os
import re
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from .context import get_config
from .pitch import detect_pitch_advanced


def _run_ffmpeg(args: List[str]) -> None:
    print(f"[ffmpeg] {' '.join(args)}")
    completed = subprocess.run(args, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg 执行失败，退出码: {completed.returncode}")


def _run_ffprobe(args: List[str]) -> Dict:
    completed = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"ffprobe 执行失败: {stderr}")
    return json.loads(completed.stdout or "{}")


def _sanitize_name(name: str) -> str:
    base = os.path.splitext(os.path.basename(name or ""))[0]
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", base)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "source"


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


def _normalize_audio_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return np.asarray(y, dtype=np.float32)


def _to_time_segments(
    intervals: np.ndarray,
    sr: int,
    min_duration: float,
) -> List[Tuple[float, float]]:
    segments: List[Tuple[float, float]] = []
    for start, end in intervals:
        start_sec = float(start) / sr
        end_sec = float(end) / sr
        if end_sec - start_sec >= min_duration:
            segments.append((start_sec, end_sec))
    return segments


def _merge_close_segments(
    segments: List[Tuple[float, float]],
    min_gap: float,
) -> List[Tuple[float, float]]:
    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= min_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _split_with_boundaries(
    interval_start: int,
    interval_end: int,
    boundaries: List[int],
    sr: int,
    min_duration: float,
) -> List[Tuple[float, float]]:
    points = [interval_start]
    points.extend(int(boundary) for boundary in boundaries if interval_start < boundary < interval_end)
    points.append(interval_end)
    points = sorted(set(points))

    segments: List[Tuple[float, float]] = []
    for start, end in zip(points[:-1], points[1:]):
        duration = (end - start) / sr
        if duration >= min_duration:
            segments.append((start / sr, end / sr))
    return segments


def _find_peak_boundaries(
    novelty: np.ndarray,
    hop_length: int,
    interval_start: int,
    sr: int,
    min_distance_sec: float,
    threshold_scale: float,
) -> List[int]:
    if novelty.size == 0:
        return []

    novelty = np.asarray(novelty, dtype=np.float32)
    max_val = float(np.max(novelty))
    if max_val <= 0:
        return []

    median = float(np.median(novelty))
    std = float(np.std(novelty))
    threshold = max(max_val * threshold_scale, median + std * 0.5)
    distance = max(1, int(min_distance_sec * sr / hop_length))
    peaks, _ = signal.find_peaks(novelty, height=threshold, distance=distance)
    return [interval_start + int(peak * hop_length) for peak in peaks]


def _frame_mask_to_intervals(
    mask: np.ndarray,
    hop_length: int,
    min_duration: float,
    sr: int,
) -> List[Tuple[float, float]]:
    if mask.size == 0:
        return []

    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    segments: List[Tuple[float, float]] = []
    for start_frame, end_frame in zip(starts, ends):
        start_sec = start_frame * hop_length / sr
        end_sec = end_frame * hop_length / sr
        if end_sec - start_sec >= min_duration:
            segments.append((start_sec, end_sec))
    return segments


def _infer_segmentation_mode(y: np.ndarray, sr: int) -> Tuple[str, Dict[str, float]]:
    hop_length = 512
    harmonic, percussive = librosa.effects.hpss(y)
    total_energy = float(np.sum(np.abs(y))) + 1e-8
    percussive_ratio = float(np.sum(np.abs(percussive)) / total_energy)
    harmonic_ratio = float(np.sum(np.abs(harmonic)) / total_energy)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    duration = max(len(y) / sr, 1e-6)
    onset_density = float(len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)) / duration)
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        hop_length=hop_length,
    )[0]
    centroid_mean = float(np.mean(spectral_centroid)) + 1e-8
    centroid_cv = float(np.std(spectral_centroid) / centroid_mean)

    voiced_ratio = 0.0
    try:
        f0, voiced_flag, _ = librosa.pyin(
            harmonic,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            hop_length=hop_length,
        )
        if voiced_flag is not None and len(voiced_flag) > 0:
            voiced_ratio = float(np.mean(voiced_flag))
    except Exception:
        pass

    if (
        voiced_ratio >= 0.45
        and percussive_ratio < 0.42
        and harmonic_ratio < 0.97
        and 0.08 <= centroid_cv <= 0.75
        and onset_density >= 0.8
    ):
        mode = "vocal"
    elif percussive_ratio >= 0.5 and onset_density >= 2.0:
        mode = "percussive"
    elif harmonic_ratio >= 0.55:
        mode = "sustained"
    else:
        mode = "percussive"

    diagnostics = {
        "percussive_ratio": percussive_ratio,
        "harmonic_ratio": harmonic_ratio,
        "onset_density": onset_density,
        "voiced_ratio": voiced_ratio,
        "centroid_cv": centroid_cv,
    }
    return mode, diagnostics


def _segment_percussive_audio(
    y: np.ndarray,
    sr: int,
    min_duration: float,
    silence_top_db: float,
) -> Tuple[List[Tuple[float, float]], Optional[str]]:
    _, percussive = librosa.effects.hpss(y)
    intervals = librosa.effects.split(
        y,
        top_db=silence_top_db,
        frame_length=2048,
        hop_length=512,
    )
    if len(intervals) == 0:
        return [], "未检测到有效有声区间"

    segments: List[Tuple[float, float]] = []
    for start, end in intervals:
        y_chunk = percussive[start:end]
        if len(y_chunk) == 0:
            continue
        onsets = librosa.onset.onset_detect(
            y=y_chunk,
            sr=sr,
            hop_length=256,
            backtrack=True,
            units="samples",
            wait=2,
        )
        if len(onsets) == 0:
            segments.append((start / sr, end / sr))
            continue
        boundaries = [start + int(onset) for onset in sorted(set(onsets))]
        segments.extend(_split_with_boundaries(start, end, boundaries, sr, min_duration))

    if not segments:
        return [], "打击型切分未得到有效片段"
    return _merge_close_segments(segments, 0.02), None


def _segment_sustained_audio(
    y: np.ndarray,
    sr: int,
    min_duration: float,
    silence_top_db: float,
) -> Tuple[List[Tuple[float, float]], Optional[str]]:
    hop_length = 512
    harmonic, _ = librosa.effects.hpss(y)
    intervals = librosa.effects.split(
        harmonic,
        top_db=silence_top_db + 12,
        frame_length=4096,
        hop_length=hop_length,
    )
    if len(intervals) == 0:
        return [], "未检测到持续音区间"

    segments: List[Tuple[float, float]] = []
    for start, end in intervals:
        y_chunk = harmonic[start:end]
        if len(y_chunk) < hop_length * 4:
            segments.append((start / sr, end / sr))
            continue

        onset_env = librosa.onset.onset_strength(
            y=y_chunk,
            sr=sr,
            hop_length=hop_length,
            aggregate=np.median,
        )
        try:
            chroma = librosa.feature.chroma_cqt(y=y_chunk, sr=sr, hop_length=hop_length)
            chroma_delta = np.linalg.norm(np.diff(chroma, axis=1), axis=0)
            chroma_delta = np.pad(chroma_delta, (1, 0))
        except Exception:
            chroma_delta = np.zeros_like(onset_env)

        if np.max(onset_env) > 0:
            onset_env = onset_env / np.max(onset_env)
        if np.max(chroma_delta) > 0:
            chroma_delta = chroma_delta / np.max(chroma_delta)

        novelty = onset_env * 0.45 + chroma_delta * 0.55
        boundaries = _find_peak_boundaries(
            novelty,
            hop_length=hop_length,
            interval_start=start,
            sr=sr,
            min_distance_sec=0.18,
            threshold_scale=0.32,
        )
        segments.extend(_split_with_boundaries(start, end, boundaries, sr, min_duration))

    if not segments:
        return [], "持续音切分未得到有效片段"
    return _merge_close_segments(segments, 0.05), None


def _segment_vocal_audio(
    y: np.ndarray,
    sr: int,
    min_duration: float,
) -> Tuple[List[Tuple[float, float]], Optional[str]]:
    hop_length = 256
    harmonic, _ = librosa.effects.hpss(y)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-6), ref=np.max)
    energy_mask = rms_db > max(-42.0, float(np.percentile(rms_db, 35)))

    voiced_mask = np.zeros_like(energy_mask, dtype=bool)
    try:
        _f0, voiced_flag, voiced_probs = librosa.pyin(
            harmonic,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            hop_length=hop_length,
        )
        if voiced_flag is not None:
            voiced_mask[: len(voiced_flag)] = voiced_flag
            if voiced_probs is not None:
                voiced_mask[: len(voiced_probs)] &= voiced_probs > 0.35
    except Exception:
        voiced_mask = energy_mask.copy()

    activity_mask = energy_mask | voiced_mask
    segments = _frame_mask_to_intervals(activity_mask, hop_length, min_duration * 1.1, sr)
    if not segments:
        return [], "未检测到稳定人声区间"

    refined: List[Tuple[float, float]] = []
    for start_sec, end_sec in segments:
        start = int(start_sec * sr)
        end = int(end_sec * sr)
        y_chunk = y[start:end]
        if len(y_chunk) < hop_length * 4:
            refined.append((start_sec, end_sec))
            continue

        onset_env = librosa.onset.onset_strength(
            y=y_chunk,
            sr=sr,
            hop_length=hop_length,
            aggregate=np.median,
        )
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y_chunk,
            sr=sr,
            hop_length=hop_length,
        )[0]
        centroid_delta = np.abs(np.diff(spectral_centroid, prepend=spectral_centroid[:1]))

        if np.max(onset_env) > 0:
            onset_env = onset_env / np.max(onset_env)
        if np.max(centroid_delta) > 0:
            centroid_delta = centroid_delta / np.max(centroid_delta)

        novelty = onset_env * 0.6 + centroid_delta * 0.4
        boundaries = _find_peak_boundaries(
            novelty,
            hop_length=hop_length,
            interval_start=start,
            sr=sr,
            min_distance_sec=0.08,
            threshold_scale=0.28,
        )
        refined.extend(_split_with_boundaries(start, end, boundaries, sr, min_duration))

    if not refined:
        return [], "人声细分未得到有效片段"
    return _merge_close_segments(refined, 0.03), None


def segment_audio(
    y: np.ndarray,
    sr: int,
    mode: str = "auto",
) -> Tuple[List[Tuple[float, float]], Optional[str], str, Dict[str, float]]:
    """按素材类型执行自动切分。"""
    config = get_config()
    y = _normalize_audio_mono(y)
    requested_mode = (mode or "auto").strip().lower()

    diagnostics: Dict[str, float] = {}
    resolved_mode = requested_mode
    if requested_mode == "auto":
        resolved_mode, diagnostics = _infer_segmentation_mode(y, sr)
    elif requested_mode not in {"percussive", "sustained", "vocal"}:
        resolved_mode = "percussive"

    if resolved_mode == "percussive":
        segments, reason = _segment_percussive_audio(
            y,
            sr,
            min_duration=config.min_clip_duration,
            silence_top_db=config.silence_threshold_db,
        )
    elif resolved_mode == "sustained":
        segments, reason = _segment_sustained_audio(
            y,
            sr,
            min_duration=max(0.08, config.min_clip_duration),
            silence_top_db=config.silence_threshold_db,
        )
    else:
        segments, reason = _segment_vocal_audio(
            y,
            sr,
            min_duration=max(0.06, config.min_clip_duration),
        )

    segments = _merge_close_segments(segments, 0.015)
    diagnostics["segment_count"] = float(len(segments))
    diagnostics["avg_duration"] = float(
        np.mean([end - start for start, end in segments]) if segments else 0.0
    )
    return segments, reason, resolved_mode, diagnostics


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


def process_video_to_clips(
    video_path: str,
    auto_segment: bool = True,
    segment_mode: str = "auto",
) -> Dict[str, object]:
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
    source_name = os.path.basename(video_path)
    source_id = f"{_sanitize_name(source_name)}_{timestamp}"
    source_video_dir = os.path.join(config.video_clip_dir, source_id)
    os.makedirs(source_video_dir, exist_ok=True)

    audio_path = os.path.join(config.video_dir, f"video_audio_{timestamp}.wav")
    extract_audio_to_wav(video_path, audio_path, config.sample_rate)

    y, sr = sf.read(audio_path)
    y = _normalize_audio_mono(y)

    resolved_mode = "full"
    diagnostics: Dict[str, float] = {}
    if auto_segment:
        segments, reason, resolved_mode, diagnostics = segment_audio(
            y,
            sr,
            mode=segment_mode,
        )
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
        seg_path = os.path.join(source_video_dir, seg_filename)
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
                "source_id": source_id,
                "source_name": source_name,
            }
        )

    return {
        "segments": results,
        "reason": reason,
        "audio_path": audio_path,
        "video_info": video_info,
        "source_id": source_id,
        "source_name": source_name,
        "segment_mode_requested": segment_mode,
        "segment_mode_resolved": resolved_mode,
        "segment_diagnostics": diagnostics,
    }
