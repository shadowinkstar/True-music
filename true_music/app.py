import json

from .clip_manager import AudioClipManager
from .context import get_config, set_clip_manager
from .fonts import setup_chinese_fonts
from .ui import build_advanced_ui


def write_default_config(config_path: str) -> None:
    config_data = {
        "audio_settings": {
            "sample_rate": 22050,
            "min_freq": 32.70,
            "max_freq": 4186.01,
            "tempo": 120,
        },
        "detection_settings": {
            "silence_threshold_db": 40,
            "confidence_threshold": 0.7,
            "min_clip_duration": 0.05,
        },
        "processing_settings": {
            "time_stretch_range": [0.5, 2.0],
            "pitch_shift_range": [-12, 12],
            "fade_in": 0.01,
            "fade_out": 0.01,
        },
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)


def main() -> None:
    setup_chinese_fonts()
    config = get_config()

    clip_manager = AudioClipManager(config)
    set_clip_manager(clip_manager)

    write_default_config(config.config_file)

    # 启动应用
    print("=" * 60)
    print("启动高级音频处理与音乐制作系统")
    print("请访问 http://localhost:7860 打开界面")
    print("=" * 60)

    app = build_advanced_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
