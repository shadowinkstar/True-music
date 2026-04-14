import argparse
import os
import sys

import librosa
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from true_music.context import get_config
from true_music.video_processing import extract_audio_to_wav, segment_audio


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _load_input_audio(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        config = get_config()
        temp_dir = os.path.join(config.output_dir, "segment_diagnostics")
        os.makedirs(temp_dir, exist_ok=True)
        temp_audio = os.path.join(temp_dir, f"{os.path.basename(path)}.wav")
        extract_audio_to_wav(path, temp_audio, config.sample_rate)
        y, sr = sf.read(temp_audio)
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        return y, sr

    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose segmentation strategy on an audio/video file.")
    parser.add_argument("input", help="Audio or video file to analyze.")
    parser.add_argument(
        "--mode",
        choices=["auto", "percussive", "sustained", "vocal"],
        default="auto",
        help="Segmentation mode.",
    )
    parser.add_argument(
        "--export-dir",
        default="",
        help="Optional directory to export segmented wav files.",
    )
    args = parser.parse_args()

    y, sr = _load_input_audio(args.input)
    segments, reason, resolved_mode, diagnostics = segment_audio(y, sr, mode=args.mode)

    print(f"requested_mode={args.mode}")
    print(f"resolved_mode={resolved_mode}")
    print(f"segment_count={len(segments)}")
    if reason:
        print(f"reason={reason}")
    if diagnostics:
        for key, value in sorted(diagnostics.items()):
            print(f"{key}={value:.4f}")

    for index, (start, end) in enumerate(segments):
        print(f"segment[{index}] start={start:.3f}s end={end:.3f}s duration={end-start:.3f}s")

    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)
        for index, (start, end) in enumerate(segments):
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment = y[start_sample:end_sample]
            output_path = os.path.join(args.export_dir, f"seg_{index:03d}.wav")
            sf.write(output_path, segment, sr)
        print(f"exported_to={args.export_dir}")


if __name__ == "__main__":
    main()
