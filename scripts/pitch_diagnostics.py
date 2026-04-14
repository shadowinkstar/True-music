import argparse
import os
import sys

import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from true_music.audio_processing import apply_fade, pitch_shift
from true_music.pitch import detect_pitch_advanced


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pitch-shift diagnostics for a single clip.")
    parser.add_argument("input", help="Path to the source audio file.")
    parser.add_argument(
        "--steps",
        type=float,
        nargs="+",
        default=[-5.0, -2.0, 2.0, 5.0],
        help="Semitone offsets to export.",
    )
    parser.add_argument(
        "--attack-ms",
        type=float,
        default=35.0,
        help="Attack window for preserve_attack mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("output", "pitch_diagnostics"),
        help="Directory for generated files.",
    )
    args = parser.parse_args()

    y, sr = sf.read(args.input)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]

    source_info = detect_pitch_advanced(y, sr)
    print(
        f"[source] note={source_info.get('note')} freq={source_info.get('frequency')} sr={sr}"
    )

    for step in args.steps:
        for mode in ("standard", "preserve_attack"):
            shifted = pitch_shift(
                y,
                sr,
                step,
                mode=mode,
                preserve_attack_ms=args.attack_ms,
            )
            shifted = apply_fade(shifted, sr, fade_in=0.005, fade_out=0.02)
            result_info = detect_pitch_advanced(shifted, sr)
            filename = f"{base_name}_{mode}_{step:+.1f}st.wav".replace("+", "p")
            filename = filename.replace("-", "m")
            output_path = os.path.join(args.output_dir, filename)
            sf.write(output_path, shifted, sr)
            print(
                "[export] "
                f"mode={mode} step={step:+.1f} "
                f"note={result_info.get('note')} "
                f"freq={result_info.get('frequency')} "
                f"path={output_path}"
            )


if __name__ == "__main__":
    main()
