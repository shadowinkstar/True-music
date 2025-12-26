import math
import re
from typing import Optional, Tuple


def note_to_midi(note: str) -> Optional[int]:
    """音名转MIDI编号，支持扩展格式"""
    if not note:
        return None

    note = note.strip()

    # 匹配格式：音名[升降号]八度（例如：C4, D#4, Gb3, A♯5, A♭5）
    pattern = r"([A-G])([#b♯♭]?)(-?\d+)"
    match = re.match(pattern, note, flags=re.IGNORECASE)

    if not match:
        return None

    note_name, accidental, octave_str = match.groups()
    note_name = note_name.upper()

    # 基本音名映射
    base_notes = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

    if note_name not in base_notes:
        return None

    midi_base = base_notes[note_name]

    # 处理升降号
    if accidental in ("#", "♯"):
        midi_base += 1
    elif accidental in ("b", "B", "♭"):
        midi_base -= 1

    try:
        octave = int(octave_str)
        midi_number = (octave + 1) * 12 + midi_base
        return midi_number
    except ValueError:
        return None


def midi_to_note(midi: int) -> str:
    """MIDI编号转音名"""
    if not 0 <= midi <= 127:
        return ""

    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = midi // 12 - 1
    note_name = notes[midi % 12]

    # 将升号替换为更易读的格式
    if "#" in note_name:
        base_note = note_name[0]
        return f"{base_note}♯{octave}"
    return f"{note_name}{octave}"


def midi_to_freq(midi: int) -> float:
    """MIDI转频率"""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def freq_to_midi(freq: float) -> float:
    """频率转MIDI（浮点数，更精确）"""
    if freq is None or not math.isfinite(freq) or freq <= 0:
        return float("nan")
    return 69 + 12 * math.log2(freq / 440.0)


def freq_to_note(freq: float) -> Tuple[str, float]:
    """频率转音名和音分偏差"""
    midi_float = freq_to_midi(freq)
    if not math.isfinite(midi_float):
        return "", 0.0
    midi_int = round(midi_float)

    # 计算音分偏差
    cents = (midi_float - midi_int) * 100

    note_name = midi_to_note(midi_int)
    return note_name, cents
