import os
import time
from typing import Dict, List
import re

import librosa
import numpy as np
import soundfile as sf

from .audio_processing import apply_fade, normalize_audio, pitch_shift
from .context import get_config, require_clip_manager
from .matching import find_best_match_for_note
from .score_parser import parse_score_notes
from .video_composition import compose_video_timeline

_last_video_context = {
    "video_timeline": None,
    "video_default_info": None,
    "audio_path": None,
}


def _compute_rms(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))


def _estimate_peak_polyphony(track_notes: List[Dict]) -> int:
    events = []
    for note in track_notes:
        if note.get("midi_pitch", -1) == -1:
            continue
        start = float(note.get("start_time", 0.0))
        end = start + float(note.get("duration", 0.0))
        events.append((start, 1))
        events.append((end, -1))

    active = 0
    peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], -item[1])):
        active += delta
        peak = max(peak, active)
    return max(peak, 1) if events else 1


def _analyze_track_mix(notes: List[Dict]) -> Dict[str, object]:
    track_notes: Dict[int, List[Dict]] = {}
    for note in notes:
        if note.get("midi_pitch", -1) == -1:
            continue
        track_num = int(note.get("track", 0) or 0)
        track_notes.setdefault(track_num, []).append(note)

    if not track_notes:
        return {"lead_track": 0, "track_stats": {}, "gain_map": {0: 1.0}}

    track_stats: Dict[int, Dict[str, float]] = {}
    for track_num, items in track_notes.items():
        avg_midi = float(np.mean([note["midi_pitch"] for note in items]))
        avg_velocity = float(np.mean([note.get("velocity", 64) for note in items]))
        peak_polyphony = _estimate_peak_polyphony(items)
        score = avg_midi + (avg_velocity / 24.0) - (peak_polyphony - 1) * 8.0
        track_stats[track_num] = {
            "avg_midi": avg_midi,
            "avg_velocity": avg_velocity,
            "peak_polyphony": float(peak_polyphony),
            "note_count": float(len(items)),
            "lead_score": score,
        }

    lead_track = max(track_stats, key=lambda track_num: track_stats[track_num]["lead_score"])
    lead_avg_midi = track_stats[lead_track]["avg_midi"]

    gain_map: Dict[int, float] = {}
    for track_num, stats in track_stats.items():
        if track_num == lead_track:
            gain_map[track_num] = 1.0
            continue

        gain = 0.62
        if stats["peak_polyphony"] > 1:
            gain -= min(0.22, (stats["peak_polyphony"] - 1) * 0.08)
        if stats["avg_midi"] < lead_avg_midi - 12:
            gain -= 0.07
        gain_map[track_num] = float(np.clip(gain, 0.35, 0.62))

    return {"lead_track": lead_track, "track_stats": track_stats, "gain_map": gain_map}


def generate_video_from_last_composition():
    """基于最近一次生成结果合成视频（便于先试听再生成视频）"""
    timeline = _last_video_context.get("video_timeline")
    video_info = _last_video_context.get("video_default_info")
    audio_path = _last_video_context.get("audio_path")

    if not timeline or not audio_path:
        return "未找到可用的生成记录，请先生成音频", None
    if not video_info:
        return "未找到包含视频信息的片段，无法生成视频", None

    try:
        output_path = compose_video_timeline(timeline, video_info, audio_path=audio_path)
        return f"✅ 视频合成完成: {os.path.basename(output_path)}", output_path
    except Exception as exc:
        return f"❌ 视频合成失败: {str(exc)}", None


def generate_music_from_clips(clip_assignments, tempo):
    """从片段生成音乐"""
    config = get_config()
    clip_manager = require_clip_manager()

    sr = config.sample_rate
    beat_duration = 60.0 / tempo

    # 解析片段分配（格式: "时间拍:片段ID,时间拍:片段ID,..."）
    assignments = []
    for assignment in clip_assignments.split(","):
        if ":" in assignment:
            beat_str, clip_id_str = assignment.split(":")
            try:
                beat = float(beat_str.strip())
                clip_id = int(clip_id_str.strip())
                assignments.append((beat, clip_id))
            except Exception:
                continue

    if not assignments:
        return "没有有效的片段分配", None

    # 按时间排序
    assignments.sort(key=lambda x: x[0])

    # 计算总时长
    last_beat = max([a[0] for a in assignments]) + 4  # 假设每个片段4拍
    total_samples = int(last_beat * beat_duration * sr)

    # 创建音轨
    track = np.zeros(total_samples)

    for beat, clip_id in assignments:
        if 0 <= clip_id < len(clip_manager.clips):
            clip = clip_manager.clips[clip_id]
            y, _ = sf.read(clip["filepath"])
            if y.ndim > 1:
                y = np.mean(y, axis=1)

            # 调整到标准时长（1拍）
            target_samples = int(beat_duration * sr)
            if len(y) > target_samples:
                y = y[:target_samples]
            else:
                y = np.pad(y, (0, target_samples - len(y)), mode="constant")

            # 添加到音轨
            start_sample = int(beat * beat_duration * sr)
            end_sample = start_sample + len(y)

            if end_sample <= len(track):
                track[start_sample:end_sample] += y

    # 归一化
    track = normalize_audio(track)

    # 保存结果
    output_filename = f"composition_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    output_path = os.path.join(config.output_dir, output_filename)
    sf.write(output_path, track, sr)

    return f"✅ 音乐生成完成: {output_filename}", (sr, track)


def _parse_source_sequence(sequence_text: str) -> List[str]:
    if not sequence_text:
        return []
    tokens = re.split(r"[,\s，;；]+", sequence_text.strip())
    return [token for token in tokens if token]


def auto_generate_music_from_score(
    score_file,
    tempo=120,
    tolerance_cents=20.0,
    use_pitch_shift=True,
    source_sequence_text="",
    allowed_video_sources_text="",
    generate_video=False,
):
    """
    自动从乐谱生成音乐的主函数
    """
    config = get_config()
    clip_manager = require_clip_manager()

    if not score_file:
        return None, "请先上传乐谱文件", [], "?? 未上传乐谱", None

    try:
        generation_status = "🚀 开始解析乐谱..."
        yield None, generation_status, [], "解析中...", None

        # 1. 解析乐谱
        notes = parse_score_notes(score_file)
        if not notes:
            return None, "❌ 未能从乐谱中解析出音符", [], "解析失败", None

        # ============ 新增：优先使用MIDI文件的原始速度 ============
        # 检查音符中是否包含原始速度信息
        original_tempos = set(n.get("tempo") for n in notes if n.get("tempo") is not None)
        if original_tempos and len(original_tempos) == 1:
            original_tempo = list(original_tempos)[0]
            if original_tempo != tempo:
                print(
                    f"[INFO] 使用MIDI文件原始速度: {original_tempo} BPM (覆盖用户设置的 {tempo} BPM)"
                )
                tempo = original_tempo
        # ==========================================================

        generation_status = (
            f"✅ 解析完成，共 {len(notes)} 个音符\n🔍 开始匹配音频片段..."
        )
        yield None, generation_status, [], "匹配中...", None

        # 2. 匹配音频片段
        sr = config.sample_rate
        beat_duration = 60.0 / tempo
        match_details = []
        video_timeline = []
        video_default_info = None
        mix_analysis = _analyze_track_mix(notes)
        lead_track = int(mix_analysis["lead_track"])
        track_gain_map = mix_analysis["gain_map"]
        source_sequence = _parse_source_sequence(source_sequence_text)
        allowed_video_sources = _parse_source_sequence(allowed_video_sources_text)
        sequence_index = 0

        # ============ 新增：计算乐谱原始理论时长（用于调试） ============
        max_beat_in_score = max([note["start_time"] + note["duration"] for note in notes])
        theory_total_seconds = max_beat_in_score * beat_duration
        print(
            f"[DEBUG_TIMING] 乐谱理论信息: 总拍数={max_beat_in_score:.2f}, tempo={tempo}, 理论时长={theory_total_seconds:.2f}秒"
        )
        # ==========================================================

        # 为每个音符匹配片段
        for i, note in enumerate(notes):
            target_midi = note["midi_pitch"]

            # >>> 修改点1：优先处理休止符 <<<
            if target_midi == -1:
                note["matched"] = True
                note["is_rest"] = True
                match_details.append(
                    [
                        f"音符{i+1}",
                        note["note_name"],
                        f"休止符 ({note['duration']:.2f}拍)",
                        "N/A",
                        "⏸️ 休止",
                        note.get("track", 0),  # 展示音轨信息
                        note.get("instrument", "rest"),
                    ]
                )
                continue  # 跳过后续匹配逻辑

            required_tag = (
                source_sequence[sequence_index % len(source_sequence)]
                if source_sequence
                else None
            )

            # 寻找最佳匹配（仅针对普通音符）
            best_clip, semitones_diff = find_best_match_for_note(
                target_midi,
                tolerance_cents,
                required_tag=required_tag,
                allowed_sources=allowed_video_sources,
            )

            if best_clip:
                note["matched"] = True
                note["clip_id"] = best_clip["id"]
                note["pitch_shift"] = semitones_diff if use_pitch_shift else 0

                match_status = (
                    "✅ 完全匹配"
                    if abs(semitones_diff) < 0.1
                    else f"🎛️ 需变调 {semitones_diff:+.1f} 半音"
                )

                match_details.append(
                    [
                        f"音符{i+1}",
                        note["note_name"],
                        f"片段{best_clip['id']} ({best_clip.get('note_info', {}).get('note', '未知')})"
                        + (f" [{required_tag}]" if required_tag else ""),
                        f"{semitones_diff:+.1f}" if use_pitch_shift else "0",
                        match_status,
                        note.get("track", 0),  # 新增：展示音轨信息
                        note.get("instrument", "unknown"),  # 新增：展示乐器信息
                    ]
                )
            else:
                note["matched"] = False
                match_details.append(
                    [
                        f"音符{i+1}",
                        note["note_name"],
                        f"无可用片段{f' (标签: {required_tag})' if required_tag else ''}",
                        "N/A",
                        "❌ 未匹配",
                        note.get("track", 0),
                        note.get("instrument", "unknown"),
                    ]
                )

            sequence_index += 1

        # 统计匹配结果（仅统计普通音符，排除休止符）
        valid_notes = [n for n in notes if n.get("midi_pitch", 0) != -1]
        matched_count = sum(1 for n in valid_notes if n["matched"])
        total_valid_notes = len(valid_notes)
        match_rate = matched_count / total_valid_notes * 100 if total_valid_notes > 0 else 0

        generation_status = (
            f"✅ 匹配完成: {matched_count}/{total_valid_notes} 个可匹配音符 ({match_rate:.1f}%)\n🛠️ 开始处理音频..."
        )
        yield None, generation_status, match_details, "处理中...", None

        # 3. 处理音频片段
        processed_clips = {}
        audio_segments = []

        for i, note in enumerate(notes):
            # ============ 新增：时间调试信息 ============
            debug_info = (
                f"音符{i}({note['note_name']}): start={note['start_time']:.2f}拍, dur={note['duration']:.2f}拍"
            )
            # ==========================================

            # >>> 修改点2：优先处理休止符 <<<
            if note.get("is_rest") or note["midi_pitch"] == -1:
                # 生成静音片段 - 关键修复：正确计算秒数
                silence_duration = note["duration"] * beat_duration  # 拍 → 秒
                silence_samples = int(silence_duration * sr)
                # 关键：存储开始时间（秒），而不是拍数
                start_time_seconds = note["start_time"] * beat_duration
                audio_segments.append(
                    (
                        start_time_seconds,
                        np.zeros(silence_samples, dtype=np.float32),
                        int(note.get("track", 0) or 0),
                    )
                )
                video_timeline.append((start_time_seconds, silence_duration, None))

                print(
                    f"[DEBUG_TIMING] {debug_info} -> 休止符: {silence_duration:.3f}秒, 开始于{start_time_seconds:.3f}秒"
                )
                continue

            # 处理未匹配的普通音符（生成静音）
            if not note["matched"]:
                silence_duration = note["duration"] * beat_duration  # 拍 → 秒
                silence_samples = int(silence_duration * sr)
                # 关键：存储开始时间（秒），而不是拍数
                start_time_seconds = note["start_time"] * beat_duration
                audio_segments.append(
                    (
                        start_time_seconds,
                        np.zeros(silence_samples, dtype=np.float32),
                        int(note.get("track", 0) or 0),
                    )
                )
                video_timeline.append((start_time_seconds, silence_duration, None))

                print(
                    f"[DEBUG_TIMING] {debug_info} -> 未匹配静音: {silence_duration:.3f}秒, 开始于{start_time_seconds:.3f}秒"
                )
                continue

            # 处理已匹配的普通音符
            clip_id = note["clip_id"]
            semitones = note["pitch_shift"]

            # 如果已处理过相同变调的片段，直接重用
            cache_key = f"{clip_id}_{semitones:.2f}"  # 固定小数位数，避免浮点误差
            if cache_key not in processed_clips:
                # 加载原始音频
                clip = clip_manager.clips[clip_id]
                y, clip_sr = sf.read(clip["filepath"])
                if y.ndim > 1:
                    y = np.mean(y, axis=1)

                # 重采样到目标采样率
                if clip_sr != sr:
                    y = librosa.resample(y, orig_sr=clip_sr, target_sr=sr)

                # 变调处理
                if use_pitch_shift and abs(semitones) > 0.1:
                    y = pitch_shift(y, sr, semitones, mode="preserve_attack")

                processed_clips[cache_key] = y

            # 获取处理后的音频
            y_processed = processed_clips[cache_key].copy()

            # 时间拉伸以匹配音符时长 - 关键：目标时长已经是秒
            target_duration = note["duration"] * beat_duration
            current_duration = len(y_processed) / sr

            print(
                f"[DEBUG_TIMING] {debug_info} -> 音频: 当前{current_duration:.3f}秒, 目标{target_duration:.3f}秒"
            )

            # 只在小范围内使用时间拉伸
            if 0.8 <= current_duration / target_duration <= 1.2:
                # 差异在±20%以内，使用时间拉伸
                rate = current_duration / target_duration
                y_processed = librosa.effects.time_stretch(y_processed, rate=rate)
                print(f"[DEBUG_TIMING] 小范围拉伸: 比率{rate:.3f}")

            # 强制匹配目标长度（裁剪或填充）
            target_samples = int(target_duration * sr)
            if len(y_processed) != target_samples:
                # 使用更智能的裁剪/填充
                if len(y_processed) > target_samples:
                    # 从中间裁剪，保持音符主体
                    start = (len(y_processed) - target_samples) // 2
                    y_processed = y_processed[start : start + target_samples]
                else:
                    # 填充静音
                    y_processed = np.pad(
                        y_processed,
                        (0, target_samples - len(y_processed)),
                        mode="constant",
                    )

            print(f"[DEBUG_TIMING] 长度调整: {len(y_processed)/sr:.3f}秒")

            # 应用音量调整（基于velocity） - 保持原MIDI响度关系
            velocity_factor = note["velocity"] / 127.0  # 标准MIDI线性映射

            # 使用线性映射，保持与原MIDI一致的响应
            # 去掉曲线调整和固定系数，让velocity直接控制增益
            y_processed *= velocity_factor

            # 添加峰值限制（防止削波，但保持相对平衡）
            max_amp = np.max(np.abs(y_processed))
            if max_amp > 1.0:  # 只在实际削波时限制
                y_processed *= 0.99 / max_amp  # 降低到99%避免削波
                print(f"[DEBUG] 音符{note.get('pitch', '?')}: 限制峰值 {max_amp:.3f} -> 0.99")

            # 先确保长度正确，再添加淡入淡出
            target_samples = int(note["duration"] * beat_duration * sr)
            if len(y_processed) != target_samples:
                if len(y_processed) > target_samples:
                    # 裁剪中间部分，保持音符主体
                    start = (len(y_processed) - target_samples) // 2
                    y_processed = y_processed[start : start + target_samples]
                else:
                    # 填充静音
                    y_processed = np.pad(
                        y_processed,
                        (0, target_samples - len(y_processed)),
                        mode="constant",
                    )

            # 添加淡入淡出（避免应用于非常短的音符）
            min_length_for_fade = int(0.05 * sr)  # 至少50ms
            if len(y_processed) > min_length_for_fade:
                # 自适应淡入淡出：短音符用较短淡出，长音符用标准淡出
                note_duration = len(y_processed) / sr

                if note_duration < 0.2:  # 短音符 (<200ms)
                    fade_in = min(0.01, note_duration * 0.1)
                    fade_out = min(0.02, note_duration * 0.2)
                else:  # 正常长度音符
                    fade_in = 0.02
                    fade_out = 0.05

                y_processed = apply_fade(
                    y_processed, sr, fade_in=fade_in, fade_out=fade_out
                )

            # 关键：存储开始时间（秒），而不是拍数
            start_time_seconds = note["start_time"] * beat_duration
            audio_segments.append((start_time_seconds, y_processed, int(note.get("track", 0) or 0)))
            clip_meta = clip_manager.clips[clip_id].get("metadata", {}) or {}
            if clip_meta.get("video_info") and not video_default_info:
                video_default_info = clip_meta.get("video_info")
            video_timeline.append(
                (
                    start_time_seconds,
                    target_duration,
                    clip_meta if clip_meta.get("video_path") else None,
                )
            )

            # 每处理10个片段更新一次状态
            if i % 10 == 0 and i > 0:
                processed_count = len(
                    [
                        n
                        for n in notes[: i + 1]
                        if not n.get("is_rest") and n["midi_pitch"] != -1
                    ]
                )
                generation_status = (
                    f"✅ 已处理 {processed_count}/{total_valid_notes} 个音符\n🛠️ 继续处理音频..."
                )
                yield None, generation_status, match_details, "处理中...", None

        generation_status = (
            f"✅ 音频处理完成，共 {len(audio_segments)} 个音频片段\n🎼 开始拼接音乐..."
        )
        yield None, generation_status, match_details, "拼接中...", None

        # 4. 拼接所有音频片段 - 关键修复：所有时间都以秒为单位
        # 计算总时长（以秒为单位）
        max_end_time_seconds = 0
        generation_status = "⏱️ 正在计算总时长..."
        yield None, generation_status, match_details, "计算时长中...", None

        for start_time_seconds, segment, _track_num in audio_segments:
            segment_duration = len(segment) / sr
            end_time_seconds = start_time_seconds + segment_duration
            if end_time_seconds > max_end_time_seconds:
                max_end_time_seconds = end_time_seconds

        print(f"[DEBUG_TIMING] 音频片段最大结束时间: {max_end_time_seconds:.2f}秒")
        print(f"[DEBUG_TIMING] 理论乐谱时长: {theory_total_seconds:.2f}秒")

        generation_status = f"✅ 总时长计算完成: {max_end_time_seconds:.2f}秒\n🧠 正在分配内存..."
        yield None, generation_status, match_details, "分配内存中...", None

        # 确保有足够的空间，加上0.5秒的余量
        total_samples = int(max_end_time_seconds * sr) + int(0.5 * sr)
        track_buffers: Dict[int, np.ndarray] = {
            int(track_num): np.zeros(total_samples, dtype=np.float32)
            for track_num in track_gain_map
        }
        final_audio = np.zeros(total_samples, dtype=np.float32)

        generation_status = f"✅ 内存分配完成: {total_samples}个样本\n📌 开始放置音频片段..."
        yield None, generation_status, match_details, "放置片段中...", None

        # 按时间线放置音频片段 - 关键：所有时间都是秒，直接乘以sr得到样本位置
        placed_count = 0
        for i, (start_time_seconds, segment, track_num) in enumerate(audio_segments):
            start_sample = int(start_time_seconds * sr)
            end_sample = start_sample + len(segment)

            # 确保片段在范围内
            if start_sample < len(final_audio):
                # 计算实际结束位置
                end_actual = min(end_sample, len(final_audio))
                # 确保段长度正确
                segment_len = end_actual - start_sample
                if segment_len > 0:
                    track_num = int(track_num)
                    if track_num not in track_buffers:
                        track_buffers[track_num] = np.zeros(total_samples, dtype=np.float32)
                    track_buffers[track_num][start_sample:end_actual] += segment[:segment_len]
                    placed_count += 1

            # 每放置10个片段更新一次状态
            if i % 10 == 0 and i > 0:
                generation_status = f"📌 已放置 {i+1}/{len(audio_segments)} 个片段..."
                yield None, generation_status, match_details, "放置片段中...", None

        generation_status = (
            f"✅ 片段放置完成: {placed_count}/{len(audio_segments)} 个片段\n🎚️ 正在按轨道混音..."
        )
        yield None, generation_status, match_details, "混音中...", None

        track_mix_results = []
        for track_num, track_audio in sorted(track_buffers.items()):
            peak_before = float(np.max(np.abs(track_audio))) if len(track_audio) else 0.0
            rms_before = _compute_rms(track_audio)
            gain = float(track_gain_map.get(track_num, 0.55))
            if peak_before > 0.95:
                track_audio *= 0.95 / peak_before
            peak_after_limit = float(np.max(np.abs(track_audio))) if len(track_audio) else 0.0
            final_audio += track_audio * gain
            track_mix_results.append(
                {
                    "track": track_num,
                    "gain": gain,
                    "peak_before": peak_before,
                    "peak_after_limit": peak_after_limit,
                    "rms_before": rms_before,
                }
            )

        generation_status = "✅ 轨道混音完成\n📈 正在归一化..."
        yield None, generation_status, match_details, "归一化中...", None

        final_audio = normalize_audio(final_audio)

        # 添加淡出效果，避免突然结束
        fade_out_samples = int(0.05 * sr)
        if fade_out_samples > 0 and fade_out_samples <= len(final_audio):
            fade_out_window = np.linspace(1, 0, fade_out_samples)
            final_audio[-fade_out_samples:] *= fade_out_window

        generation_status = "✅ 音频处理完成\n📝 正在生成报告..."
        yield None, generation_status, match_details, "生成报告中...", None

        # 5. 生成报告
        actual_duration = len(final_audio) / sr
        report = f"""
        ## 📝 音乐生成报告

        ### 基本信息
        - **乐谱文件**: {os.path.basename(score_file)}
        - **音符总数**: {len(notes)} (含休止符)
        - **可匹配音符**: {total_valid_notes} (不含休止符)
        - **演奏速度**: {tempo} BPM
        - **来源序列**: {" -> ".join(source_sequence) if source_sequence else "未启用"}
        - **理论时长**: {theory_total_seconds:.2f} 秒
        - **实际生成时长**: {actual_duration:.2f} 秒
        - **时长比例**: {actual_duration/theory_total_seconds*100:.1f}%
        - **采样率**: {sr} Hz

        ### 匹配情况
        - **成功匹配**: {matched_count} 个可匹配音符 ({match_rate:.1f}%)
        - **需要变调**: {sum(1 for n in valid_notes if n['matched'] and abs(n.get('pitch_shift', 0)) > 0.1)} 个
        - **未匹配**: {total_valid_notes - matched_count} 个
        - **休止符**: {len(notes) - total_valid_notes} 个

        ### 音频处理
        - **生成的片段**: {len(audio_segments)} 个
        - **成功放置**: {placed_count} 个片段
        - **峰值电平**: {np.max(np.abs(final_audio)):.3f}

        ### 使用片段
        """

        # 统计使用的片段
        used_clips = {}
        for note in valid_notes:
            if note["matched"]:
                clip_id = note["clip_id"]
                used_clips[clip_id] = used_clips.get(clip_id, 0) + 1

        for clip_id, count in used_clips.items():
            clip = clip_manager.clips[clip_id]
            note_name = clip.get("note_info", {}).get("note", "未知")
            report += f"- **片段{clip_id}** ({note_name}): 使用 {count} 次\n"

        # >>> 修改点3：添加音轨与乐器统计 <<<
        report += "\n### 音轨与乐器信息\n"
        # 统计音轨
        tracks_used = set(
            n.get("track", 0) for n in notes if n.get("track") is not None
        )
        report += f"- **使用音轨数**: {len(tracks_used)} 个\n"
        report += f"- **自动识别主音轨**: 轨道{lead_track}\n"

        # 按音轨统计音符
        if len(tracks_used) > 1:
            report += "- **各音轨音符分布**:\n"
            for track_num in sorted(tracks_used):
                track_notes = [
                    n
                    for n in notes
                    if n.get("track", 0) == track_num and n["midi_pitch"] != -1
                ]
                if track_notes:
                    instr = track_notes[0].get("instrument", "unknown")
                    report += f"  - 音轨{track_num} ({instr}): {len(track_notes)} 个音符\n"

        if track_mix_results:
            report += "- **各音轨混音参数**:\n"
            for track_result in track_mix_results:
                track_num = track_result["track"]
                track_stat = mix_analysis["track_stats"].get(track_num, {})
                report += (
                    "  - "
                    f"音轨{track_num}: gain={track_result['gain']:.2f}, "
                    f"peak={track_result['peak_before']:.3f}->{track_result['peak_after_limit']:.3f}, "
                    f"rms={track_result['rms_before']:.3f}, "
                    f"avg_midi={track_stat.get('avg_midi', 0.0):.1f}, "
                    f"poly={track_stat.get('peak_polyphony', 1.0):.0f}\n"
                )

        # 统计乐器（仅统计非休止符）
        instruments_used = {}
        for note in valid_notes:
            instr = note.get("instrument", "unknown")
            instruments_used[instr] = instruments_used.get(instr, 0) + 1

        if instruments_used:
            report += "- **乐器分布**:\n"
            for instr, count in sorted(
                instruments_used.items(), key=lambda x: x[1], reverse=True
            ):
                report += f"  - {instr}: {count} 个音符\n"

        report += "\n### 调试信息\n"
        report += f"- **最大结束时间**: {max_end_time_seconds:.2f} 秒\n"
        report += f"- **总样本数**: {total_samples} 个\n"
        report += f"- **实际音频时长**: {actual_duration:.2f} 秒\n"

        # 显示原始速度信息（如果有）
        if "original_tempo" in locals():
            report += f"- **原始乐谱速度**: {original_tempo:.0f} BPM\n"
        report += f"- **实际使用速度**: {tempo} BPM\n"

        report += f"\n⏱️ **生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}"

        # 保存生成的音乐
        output_filename = f"auto_composition_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        output_path = os.path.join(config.output_dir, output_filename)
        sf.write(output_path, final_audio, sr)
        composition_video = None

        # 缓存最近一次生成结果，供后续视频合成
        _last_video_context["video_timeline"] = video_timeline
        _last_video_context["video_default_info"] = video_default_info
        _last_video_context["audio_path"] = output_path
        if generate_video and video_timeline and video_default_info:
            try:
                composition_video = compose_video_timeline(
                    video_timeline, video_default_info, audio_path=output_path
                )
                report += (
                    f"\n### 视频合成\n- **输出视频**: {os.path.basename(composition_video)}\n"
                )
            except Exception as exc:
                report += f"\n### 视频合成\n- **失败原因**: {str(exc)}\n"
        elif generate_video:
            report += "\n### 视频合成\n- **失败原因**: 未找到包含视频信息的片段\n"

        generation_status = f"? 音乐生成完成！\n?? 已保存至: {output_filename}"

        yield (sr, final_audio), report, match_details, generation_status, composition_video

    except Exception as exc:
        error_msg = f"❌ 生成过程中出错: {str(exc)}"
        print(f"生成音乐失败: {exc}")
        import traceback

        traceback.print_exc()
        yield None, error_msg, [], "生成失败", None
        
