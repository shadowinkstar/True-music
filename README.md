TRUE MUSIC
==========
Audio analysis, clip management, and score-driven composition in one UI.


WHY IT EXISTS
------------
This project turns raw audio clips into a playable palette, then matches them
against MusicXML or MIDI scores to generate full compositions. It is built
for iteration: quick uploads, fast matching, and a Gradio UI that keeps every
stage visible.


FEATURES
--------
- Pitch detection with multiple algorithms for better stability.
- Clip library with indexing for fast note matching.
- Time-stretching and pitch-shifting with sensible safety bounds.
- Score parsing for MusicXML and MIDI (multi-track + rests).
- Rich visual analysis: spectrograms, waveform, centroid, and ZCR.
- One-click auto composition with detailed generation reports.


PROJECT LAYOUT
--------------
Entry point stays in `main.py`, logic lives in `true_music/`:

```
true_music/
  app.py             # startup, font setup, config file generation
  ui.py              # Gradio UI and callbacks
  music_generation.py# score -> audio pipeline
  score_parser.py    # MusicXML + MIDI parsing
  matching.py        # note-to-clip matching logic
  clip_manager.py    # clip storage and index cache
  pitch.py           # pitch detection
  audio_processing.py# time-stretch, pitch-shift, fades
  visualization.py   # plots
  theory.py          # MIDI and note utilities
  config.py          # AppConfig
  context.py         # shared app state
```


QUICK START
-----------
1) Create a virtual environment and install deps:

```bash
pip install gradio librosa numpy soundfile matplotlib scipy
```

2) Run the app:

```bash
python main.py
```

3) Open the UI:
`http://localhost:7860`


DATA FOLDERS
------------
- `data/clips/` stores recorded/processed clips.
- `data/clips.json` stores clip metadata.
- `output/` stores generated compositions.


CONFIG
------
At startup, a `config.json` file is written with default settings:
- sample rate, detection thresholds, and processing ranges.

If you want to change defaults, edit `true_music/config.py` and
restart the app.


TROUBLESHOOTING
---------------
- Chinese labels show as boxes: install a Chinese font on your OS.
- No pitch detected: try cleaner input or increase clip duration.
- Mismatched timing: ensure your score is clean and the tempo is correct.


BRANCH WORKFLOW
---------------
- `dev` is the daily integration branch. Treat it as the branch that should always contain the latest usable combined work.
- Local feature work should branch from `dev`, then merge back into `dev`.
- Codex-generated branches should also be merged back into `dev` after review or verification.
- `main` stays for relatively stable milestones. Promote changes from `dev` to `main` only when the result is ready.

Recommended flow:

```bash
git switch dev
git pull origin dev

# start work
git switch -c feature/your-change

# after finishing
git switch dev
git merge feature/your-change
git push origin dev
```

For Codex branches:

```bash
git switch dev
git pull origin dev
git merge origin/codex/<branch-name>
git push origin dev
```

Current project convention:
- `dev` is the source of truth for active development.
- Both local changes and future Codex changes should end up on `dev`.
- `main` is updated from `dev`, not the other way around, unless there is a deliberate hotfix.


LICENSE
-------
Add your preferred license here.
