import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from .context import get_clip_manager
from .pitch import detect_pitch_advanced
from .serialization import convert_to_serializable
from .theory import freq_to_midi, note_to_midi

_clip_index_cache = None  # 缓存索引，避免每次重建


def build_clip_index():
    """
    构建音频片段的音高索引，加速查找。
    返回结构: {rounded_midi: [(clip_info, original_midi, confidence), ...]}
    """
    global _clip_index_cache
    if _clip_index_cache is not None:
        return _clip_index_cache

    manager = get_clip_manager()
    if manager is None:
        return {}

    index = {}
    available_clips = manager.get_all_clips()

    for clip in available_clips:
        note_info = clip.get("note_info", {})
        if note_info and note_info.get("frequency") and note_info.get("confidence"):
            clip_freq = note_info["frequency"]
            clip_midi = freq_to_midi(clip_freq)  # 精确的浮点数MIDI
            confidence = note_info.get("confidence", 0.5)

            # 将MIDI音高四舍五入到最接近的整数，作为索引键
            rounded_midi = int(round(clip_midi))

            if rounded_midi not in index:
                index[rounded_midi] = []

            index[rounded_midi].append(
                {
                    "clip": clip,
                    "exact_midi": clip_midi,  # 保存精确值用于计算
                    "confidence": confidence,
                    # 可以在这里扩展存储 instrument_tag 等元数据
                }
            )

    _clip_index_cache = index
    print(
        f"[索引构建完成] 共 {len(available_clips)} 个片段，索引到 {len(index)} 个不同音高。"
    )
    return index


def clear_clip_index_cache():
    """当添加或删除音频片段后，调用此函数清除索引缓存"""
    global _clip_index_cache
    _clip_index_cache = None
    print("音频片段索引缓存已清除，将在下次匹配时重建。")


class AudioClipManager:
    """音频片段管理器"""

    def __init__(self, config):
        self.config = config
        self.clips = []
        self.load_clips()

    def load_clips(self):
        """加载已保存的片段"""
        if os.path.exists(self.config.clip_data_file):
            with open(self.config.clip_data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.clips = data.get("clips", [])

    def save_clips(self):
        """保存片段信息"""
        clips_serializable = convert_to_serializable(self.clips)
        data = {"clips": clips_serializable}
        with open(self.config.clip_data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_clip(
        self, audio_data: np.ndarray, sr: int, note_info: Dict = None, metadata: Dict = None
    ) -> str:
        """添加音频片段"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"clip_{timestamp}.wav"
        filepath = os.path.join(self.config.clip_dir, filename)

        # 保存音频文件
        sf.write(filepath, audio_data, sr)

        # 如果没有提供音高信息，则检测
        if note_info is None:
            note_info = detect_pitch_advanced(audio_data, sr)

        # 创建片段信息
        clip_info = {
            "id": int(len(self.clips)),
            "filename": filename,
            "filepath": filepath,
            "sample_rate": int(sr),
            "duration": float(len(audio_data) / sr),
            "note_info": convert_to_serializable(note_info) if note_info else {},
            "metadata": convert_to_serializable(metadata) if metadata else {},
            "created_at": timestamp,
            "tags": [],
        }

        self.clips.append(clip_info)
        self.save_clips()
        clear_clip_index_cache()
        return clip_info

    def get_clip_by_note(
        self, target_note: str, tolerance_cents: float = 50
    ) -> List[Dict]:
        """根据音名获取片段"""
        target_midi = note_to_midi(target_note)
        if target_midi is None:
            return []

        matching_clips = []
        for clip in self.clips:
            note_info = clip.get("note_info", {})
            if note_info.get("note") and note_info.get("frequency"):
                clip_midi = freq_to_midi(note_info["frequency"])
                cents_diff = (clip_midi - target_midi) * 100

                if abs(cents_diff) <= tolerance_cents:
                    matching_clips.append({**clip, "cents_diff": cents_diff})

        # 按偏差排序
        matching_clips.sort(key=lambda x: abs(x["cents_diff"]))
        return matching_clips

    def get_all_clips(self) -> List[Dict]:
        """获取所有片段"""
        return self.clips

    def delete_clip(self, clip_id: int) -> bool:
        """删除片段，包括json记录和音频文件"""
        if 0 <= clip_id < len(self.clips):
            clip = self.clips.pop(clip_id)

            # 尝试删除音频文件
            try:
                if os.path.exists(clip["filepath"]):
                    os.remove(clip["filepath"])
                    print(f"已删除音频文件: {clip['filepath']}")
            except Exception as exc:
                print(f"删除音频文件时出错 {clip['filepath']}: {exc}")

            # 更新ID
            for i, clip_item in enumerate(self.clips):
                clip_item["id"] = i

            self.save_clips()
            clear_clip_index_cache()
            return True
        return False

    def cleanup_orphaned_files(self):
        """清理没有对应记录的音频文件"""
        # 获取所有应该存在的文件路径
        expected_files = {clip["filepath"] for clip in self.clips}

        # 遍历clips目录
        clips_dir = self.config.clip_dir
        for filename in os.listdir(clips_dir):
            if filename.endswith(".wav"):
                filepath = os.path.join(clips_dir, filename)
                if filepath not in expected_files:
                    try:
                        os.remove(filepath)
                        print(f"清理孤立文件: {filename}")
                    except Exception as exc:
                        print(f"清理文件失败 {filename}: {exc}")

    def delete_all_clips(self):
        """删除所有片段和对应的文件"""
        deleted_files = []
        for clip in self.clips:
            try:
                if os.path.exists(clip["filepath"]):
                    os.remove(clip["filepath"])
                    deleted_files.append(clip["filename"])
            except Exception as exc:
                print(f"删除文件失败 {clip['filename']}: {exc}")

        self.clips = []
        self.save_clips()
        clear_clip_index_cache()
        return deleted_files
