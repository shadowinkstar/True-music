from typing import Optional, Tuple

from .clip_manager import build_clip_index
from .context import get_clip_manager
from .theory import freq_to_midi


def find_best_match_for_note(
    target_midi: int, tolerance_cents: float = 50.0, use_confidence_weight: bool = True
) -> Tuple[Optional[dict], float]:
    """
    为目标音符寻找最佳匹配的音频片段（优化版）。

    参数:
        target_midi: 目标MIDI音高 (整数，如 60 代表 C4)
        tolerance_cents: 音高容差 (音分)
        use_confidence_weight: 是否使用置信度作为权重

    返回:
        (最佳片段信息, 需要变调的半音数)
    """
    # 1. 获取或构建索引
    index = build_clip_index()
    if not index:
        return None, 0.0  # 无可用片段

    best_clip = None
    best_semitones = 0.0
    best_score = -float("inf")  # 使用评分系统，分数越高越好

    # 2. 确定搜索范围：目标音高附近 ± (容差/100 + 1) 个半音
    search_semitones = int(tolerance_cents / 100) + 2
    lower_bound = target_midi - search_semitones
    upper_bound = target_midi + search_semitones

    # 3. 在索引的邻近键中搜索
    for search_midi in range(lower_bound, upper_bound + 1):
        if search_midi not in index:
            continue

        for clip_data in index[search_midi]:
            clip = clip_data["clip"]
            clip_exact_midi = clip_data["exact_midi"]
            confidence = clip_data["confidence"]

            # 计算精确的音高差异（半音）
            semitones_diff = target_midi - clip_exact_midi
            cents_diff = semitones_diff * 100.0

            # 如果在绝对容差范围内，才考虑
            if abs(cents_diff) <= tolerance_cents:
                # 计算匹配分数：音分越接近、置信度越高，分数越高
                closeness_score = 1.0 - (abs(cents_diff) / tolerance_cents)  # 0到1
                confidence_score = confidence if use_confidence_weight else 1.0

                # 综合分数 (可以调整权重)
                total_score = (closeness_score * 0.7) + (confidence_score * 0.3)

                if total_score > best_score:
                    best_score = total_score
                    best_clip = clip
                    best_semitones = semitones_diff

    # 4. 如果未找到容差内的，返回最接近的（原逻辑的降级方案）
    if best_clip is None:
        print(
            f"[匹配警告] 未在容差 {tolerance_cents} 音分内找到 MIDI {target_midi} 的匹配，返回最接近的。"
        )
        # 简单实现：遍历所有片段找最接近的
        manager = get_clip_manager()
        if manager is None:
            return None, 0.0
        available_clips = manager.get_all_clips()
        best_distance = float("inf")
        for clip in available_clips:
            note_info = clip.get("note_info", {})
            if note_info and note_info.get("frequency"):
                clip_freq = note_info["frequency"]
                clip_midi = freq_to_midi(clip_freq)
                semitones_diff = target_midi - clip_midi
                cents_diff = abs(semitones_diff * 100)
                if cents_diff < best_distance:
                    best_distance = cents_diff
                    best_clip = clip
                    best_semitones = semitones_diff

    return best_clip, best_semitones
