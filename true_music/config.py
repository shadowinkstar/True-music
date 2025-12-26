import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AppConfig:
    # 音频处理参数
    sample_rate: int = 22050
    min_freq: float = 32.70  # C1
    max_freq: float = 4186.01  # C8
    tempo: int = 120  # BPM
    beat_duration: float = 60.0 / 120  # 一拍的时间（秒）
    beat_division: int = 4  # 每拍分割数（4表示16分音符）
    time_stretch_range: Tuple[float, float] = (0.5, 2.0)  # 时间拉伸范围
    pitch_shift_range: Tuple[int, int] = (-12, 12)  # 音高移动范围（半音）

    # 检测参数
    silence_threshold_db: float = 40
    min_clip_duration: float = 0.05
    confidence_threshold: float = 0.7

    # 文件路径
    output_dir: str = "output"
    config_file: str = "config.json"
    # 文件路径配置
    data_dir: str = "data"  # 数据根目录
    clip_subdir: str = "clips"  # 片段子目录
    clip_data_filename: str = "clips.json"  # 片段数据文件名

    @property
    def clip_dir(self) -> str:
        """获取片段音频目录"""
        return os.path.join(self.data_dir, self.clip_subdir)

    @property
    def clip_data_file(self) -> str:
        """获取片段数据文件路径"""
        return os.path.join(self.data_dir, self.clip_data_filename)

    def __post_init__(self):
        """初始化后确保目录存在"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.clip_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
