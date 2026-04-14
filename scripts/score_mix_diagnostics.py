import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from true_music.music_generation import _analyze_track_mix
from true_music.score_parser import parse_score_notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect track-role and gain heuristics for a score.")
    parser.add_argument("score", help="Path to a MIDI or MusicXML file.")
    args = parser.parse_args()

    notes = parse_score_notes(args.score)
    if not notes:
        print("No notes parsed.")
        return

    analysis = _analyze_track_mix(notes)
    print(f"lead_track={analysis['lead_track']}")
    for track_num, stats in sorted(analysis["track_stats"].items()):
        gain = analysis["gain_map"].get(track_num, 1.0)
        print(
            f"track={track_num} "
            f"gain={gain:.2f} "
            f"avg_midi={stats['avg_midi']:.2f} "
            f"avg_velocity={stats['avg_velocity']:.2f} "
            f"peak_polyphony={stats['peak_polyphony']:.0f} "
            f"notes={stats['note_count']:.0f} "
            f"lead_score={stats['lead_score']:.2f}"
        )


if __name__ == "__main__":
    main()
