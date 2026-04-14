import librosa
import numpy as np

from .context import get_config


def _fit_audio_length(y: np.ndarray, target_samples: int) -> np.ndarray:
    """Pad or trim audio to an exact number of samples."""
    if target_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    if len(y) > target_samples:
        return y[:target_samples]
    if len(y) < target_samples:
        return np.pad(y, (0, target_samples - len(y)), mode="constant")
    return y


def time_stretch(y: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
    """
    时间拉伸（变速不变调）
    """
    config = get_config()
    current_duration = len(y) / sr
    rate = current_duration / target_duration

    # 限制拉伸范围
    rate = np.clip(rate, *config.time_stretch_range)

    # 使用librosa的时间拉伸
    y_stretched = librosa.effects.time_stretch(y, rate=rate)

    # 确保长度准确
    target_samples = int(target_duration * sr)
    if len(y_stretched) > target_samples:
        y_stretched = y_stretched[:target_samples]
    else:
        y_stretched = np.pad(
            y_stretched,
            (0, target_samples - len(y_stretched)),
            mode="constant",
        )

    return y_stretched


def pitch_shift(
    y: np.ndarray,
    sr: int,
    semitones: float,
    mode: str = "standard",
    preserve_attack_ms: float = 35.0,
) -> np.ndarray:
    """
    音高平移（变调不变速）
    """
    config = get_config()
    # 限制移动范围
    semitones = np.clip(semitones, *config.pitch_shift_range)
    if len(y) == 0 or abs(semitones) < 1e-6:
        return np.asarray(y, dtype=np.float32)

    shifted = librosa.effects.pitch_shift(
        y,
        sr=sr,
        n_steps=semitones,
        bins_per_octave=12,
    )
    shifted = _fit_audio_length(np.asarray(shifted, dtype=np.float32), len(y))

    if mode != "preserve_attack":
        return shifted

    # Preserve the original attack so short syllables keep their consonant/color.
    attack_samples = int(max(0.0, preserve_attack_ms) * sr / 1000.0)
    attack_samples = min(attack_samples, max(0, len(y) - 1))
    if attack_samples <= 0:
        return shifted

    output = shifted.copy()
    output[:attack_samples] = np.asarray(y[:attack_samples], dtype=np.float32)

    crossfade_samples = min(int(0.015 * sr), len(y) - attack_samples)
    if crossfade_samples > 1:
        fade = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)
        start = attack_samples
        end = attack_samples + crossfade_samples
        output[start:end] = (
            np.asarray(y[start:end], dtype=np.float32) * (1.0 - fade)
            + shifted[start:end] * fade
        )

    return output.astype(np.float32)


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """音频归一化"""
    if len(y) == 0:
        return y

    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        return y / max_amp * 0.9  # 留一点headroom
    return y


def apply_fade(
    y: np.ndarray, sr: int, fade_in: float = 0.01, fade_out: float = 0.01
) -> np.ndarray:
    """应用淡入淡出"""
    if len(y) == 0:
        return y

    fade_in_samples = int(fade_in * sr)
    fade_out_samples = int(fade_out * sr)

    # 创建淡入淡出窗口
    if fade_in_samples > 0:
        fade_in_window = np.linspace(0, 1, fade_in_samples)
        if fade_in_samples <= len(y):
            y[:fade_in_samples] *= fade_in_window

    if fade_out_samples > 0:
        fade_out_window = np.linspace(1, 0, fade_out_samples)
        if fade_out_samples <= len(y):
            y[-fade_out_samples:] *= fade_out_window

    return y
