from typing import Any, Dict, Optional

import librosa
import numpy as np
from scipy import signal

from .context import get_config
from .theory import freq_to_note


def detect_pitch_advanced(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    高级音高检测，返回详细信息
    """
    config = get_config()

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # 裁剪静音
    y_trimmed, _ = librosa.effects.trim(
        y,
        top_db=config.silence_threshold_db,
        frame_length=2048,
        hop_length=512,
    )

    if len(y_trimmed) < sr * config.min_clip_duration:
        return {
            "frequency": None,
            "note": None,
            "cents": 0,
            "confidence": 0,
            "stable": False,
        }

    # 使用多种方法检测音高
    freqs = []
    confidences = []

    # 方法1: YIN算法
    f0_yin = librosa.yin(
        y_trimmed,
        fmin=config.min_freq,
        fmax=config.max_freq,
        sr=sr,
        frame_length=2048,
        hop_length=512,
    )
    f0_yin = f0_yin[f0_yin > 0]
    if len(f0_yin) > 0:
        freqs.append(np.median(f0_yin))
        # 用稳定度作为置信度
        confidences.append(
            1.0 - (np.std(f0_yin) / np.mean(f0_yin)) if np.mean(f0_yin) > 0 else 0
        )

    # 方法2: PYIN算法（更准确但更慢）
    try:
        f0_pyin, voiced_flag, voiced_probs = librosa.pyin(
            y_trimmed,
            fmin=config.min_freq,
            fmax=config.max_freq,
            sr=sr,
        )
        f0_pyin = f0_pyin[voiced_flag]
        if len(f0_pyin) > 0:
            freqs.append(np.median(f0_pyin))
            confidences.append(np.mean(voiced_probs[voiced_flag]))
    except Exception:
        pass

    # 方法3: 谐波乘积谱（对音乐信号更准确）
    try:
        f0_hps = _detect_pitch_hps(y_trimmed, sr)
        if f0_hps:
            freqs.append(f0_hps)
            confidences.append(0.7)
    except Exception:
        pass

    if not freqs:
        return {
            "frequency": None,
            "note": None,
            "cents": 0,
            "confidence": 0,
            "stable": False,
        }

    # 加权平均
    weights = np.array(confidences)
    if weights.sum() == 0:
        weights = np.ones(len(freqs))

    weighted_freq = np.average(freqs, weights=weights)
    avg_confidence = np.mean(confidences)

    # 转换为音名
    note_name, cents = freq_to_note(weighted_freq)

    # 判断是否稳定
    is_stable = avg_confidence > config.confidence_threshold and len(y_trimmed) > sr * 0.2

    return {
        "frequency": float(weighted_freq) if weighted_freq is not None else None,
        "note": note_name,
        "cents": float(cents) if cents is not None else None,
        "confidence": float(avg_confidence),
        "stable": bool(is_stable),
        "duration": float(len(y_trimmed) / sr),
    }


def _detect_pitch_hps(y: np.ndarray, sr: int) -> Optional[float]:
    """谐波乘积谱音高检测"""
    config = get_config()
    n_fft = 2048
    hop_length = 512

    # 计算频谱
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # 谐波乘积谱
    hps = S.copy()
    for harmonic in (2, 3, 4):
        downsampled = signal.resample_poly(S, 1, harmonic, axis=0)
        hps = hps[: downsampled.shape[0]] * downsampled

    # 寻找峰值
    hps_mean = np.mean(hps, axis=1)
    peaks, properties = signal.find_peaks(hps_mean, height=np.max(hps_mean) * 0.1)

    if len(peaks) == 0:
        return None

    # 找到最低频率的峰值（基频）
    min_peak_idx = peaks[np.argmin(peaks)]
    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)[min_peak_idx]

    # 限制在合理范围内
    if config.min_freq <= freq <= config.max_freq:
        return freq
    return None
