import time

import gradio as gr
import numpy as np
import soundfile as sf

from .audio_processing import apply_fade
from .context import require_clip_manager
from .music_generation import auto_generate_music_from_score, generate_music_from_clips
from .pitch import detect_pitch_advanced
from .score_parser import parse_score_notes
from .serialization import convert_to_serializable
from .visualization import create_enhanced_analysis, create_spectrogram


def handle_audio_upload(audio_input, target_note, auto_detect, analysis_mode):
    """处理音频上传"""
    clip_manager = require_clip_manager()

    if audio_input is None:
        return "请先上传音频文件", None, None, None

    # 读取音频
    if isinstance(audio_input, tuple):
        sr, y = audio_input
        y = np.array(y, dtype=np.float32)
    elif isinstance(audio_input, dict):
        sr, y = audio_input["sample_rate"], np.array(audio_input["data"], dtype=np.float32)
    elif isinstance(audio_input, str):
        y, sr = sf.read(audio_input)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
    else:
        return "不支持的音频格式", None, None, None

    # 检测音高
    note_info = detect_pitch_advanced(y, sr)

    message = []

    if note_info["frequency"]:
        message.append(f"检测到频率: **{note_info['frequency']:.1f} Hz**")
        message.append(f"音名: **{note_info['note']}**")
        message.append(f"音分偏差: **{note_info['cents']:+.1f} cents**")
        message.append(f"置信度: **{note_info['confidence']:.2%}**")

        if note_info["stable"]:
            message.append("✅ 音高稳定")
        else:
            message.append("⚠️ 音高不稳定，可能包含滑音或多音")
    else:
        message.append("❌ 无法检测到稳定音高")

    # 如果有目标音高，进行比较
    if target_note:
        from .theory import midi_to_freq, note_to_midi, freq_to_midi

        target_midi = note_to_midi(target_note)
        if target_midi:
            target_freq = midi_to_freq(target_midi)
            if note_info["frequency"]:
                cents_diff = (freq_to_midi(note_info["frequency"]) - target_midi) * 100
                message.append(f"目标音高: **{target_note}** ({target_freq:.1f} Hz)")
                message.append(f"偏差: **{cents_diff:+.1f} cents**")

                if abs(cents_diff) <= 50:
                    message.append("✅ 在可接受范围内 (±50 cents)")
                else:
                    message.append("⚠️ 偏差较大")
        else:
            message.append(f"❌ 目标音高 '{target_note}' 格式错误")

    # 保存片段
    clip_info = clip_manager.add_clip(
        y,
        sr,
        note_info=convert_to_serializable(note_info) if note_info else None,
        metadata={
            "target_note": str(target_note) if target_note else "",
            "upload_time": str(time.strftime("%Y-%m-%d %H:%M:%S")),
        },
    )

    # 生成图表
    if analysis_mode == "simple":
        fig = create_spectrogram(y, sr, note_info.get("frequency"))
        fig2 = None
    else:
        fig = create_spectrogram(y, sr, note_info.get("frequency"))
        fig2 = create_enhanced_analysis(y, sr, note_info)

    return "\n".join(message), clip_info["id"], fig, fig2


def process_audio_clip(clip_id, operation, value):
    """处理音频片段（变速/变调）"""
    clip_manager = require_clip_manager()

    if not 0 <= clip_id < len(clip_manager.clips):
        return "无效的片段ID", None

    clip = clip_manager.clips[clip_id]
    y, sr = sf.read(clip["filepath"])
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if operation == "time_stretch":
        target_duration = float(value)
        from .audio_processing import time_stretch

        y_processed = time_stretch(y, sr, target_duration)
        message = f"时长调整为 {target_duration:.2f} 秒"
    elif operation == "pitch_shift":
        semitones = float(value)
        from .audio_processing import pitch_shift

        y_processed = pitch_shift(y, sr, semitones)
        message = f"音高调整 {semitones:+.1f} 个半音"
    else:
        return "未知操作", None

    # 应用淡入淡出
    y_processed = apply_fade(y_processed, sr)

    # 保存处理后的音频
    processed_info = clip_manager.add_clip(
        y_processed,
        sr,
        metadata={
            "original_clip_id": clip_id,
            "operation": operation,
            "value": value,
            "processed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    return f"✅ {message} (新片段ID: {processed_info['id']})", (sr, y_processed)


def build_music_composition_tab():
    """构建全新的自动音乐制作界面"""

    with gr.TabItem("🎼 智能音乐制作"):
        gr.Markdown(
            """
        ## 🎼 智能音乐制作工作台
        上传乐谱 → 自动匹配音频片段 → 智能变调处理 → 生成完整音乐
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # 乐谱上传区域
                gr.Markdown("### 1. 上传乐谱")
                score_upload = gr.File(
                    label="选择乐谱文件",
                    file_types=[".xml", ".musicxml", ".mid", ".midi"],
                    type="filepath",
                )

                # 乐谱信息展示
                score_info = gr.Markdown("等待上传乐谱...", label="乐谱信息")

                # 处理选项
                gr.Markdown("### 2. 处理选项")

                with gr.Row():
                    match_tolerance = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=20,
                        step=5,
                        label="音高匹配容差 (cents)",
                        info="值越小匹配要求越严格",
                    )

                    use_pitch_shift = gr.Checkbox(
                        label="启用智能变调",
                        value=True,
                        info="对不匹配的音符自动变调处理",
                    )

                tempo_input = gr.Slider(
                    label="演奏速度 (BPM)",
                    minimum=40,
                    maximum=240,
                    value=120,
                    step=5,
                )

                # 生成按钮
                btn_generate = gr.Button("🎵 自动生成音乐", variant="primary", size="lg")
                generation_status = gr.Markdown("准备生成...", label="生成状态")

            with gr.Column(scale=2):
                # 生成结果区域
                gr.Markdown("### 3. 生成结果")

                with gr.Tabs():
                    with gr.TabItem("🎧 试听音乐"):
                        composition_audio = gr.Audio(label="生成音乐", type="numpy")

                    with gr.TabItem("📊 生成报告"):
                        generation_report = gr.Markdown("生成报告将在此显示...", label="详细报告")

                    with gr.TabItem("🧩 音符匹配详情"):
                        notes_match_table = gr.Dataframe(
                            headers=["序号", "音名", "匹配片段", "变调(半音)", "状态", "音轨", "乐器"],
                            label="音符匹配情况",
                            datatype=["str", "str", "str", "str", "str", "str", "str"],
                            row_count=10,
                            interactive=False,
                        )

        # 连接生成按钮
        btn_generate.click(
            fn=auto_generate_music_from_score,
            inputs=[score_upload, tempo_input, match_tolerance, use_pitch_shift],
            outputs=[composition_audio, generation_report, notes_match_table, generation_status],
        )

        # 乐谱上传后的预览
        def preview_score(filepath):
            if filepath is None:
                return "等待上传乐谱...", []

            try:
                notes = parse_score_notes(filepath)
                if not notes:
                    return "未能从乐谱中解析出音符", []

                # 构建预览信息
                preview_text = "### 乐谱解析成功！\n"
                preview_text += f"**音符总数**: {len(notes)}\n"
                preview_text += f"**音高范围**: {notes[0]['note_name']} 到 {notes[-1]['note_name']}\n"
                preview_text += f"**总时长**: {sum(n['duration'] for n in notes):.2f} 拍\n\n"
                preview_text += "**前10个音符**:\n"

                # 构建表格数据
                table_data = []
                for i, note in enumerate(notes[:10]):
                    table_data.append(
                        [
                            i + 1,
                            note["note_name"],
                            f"{note['duration']:.2f}拍",
                            f"{note['start_time']:.2f}拍",
                            "是" if note["matched"] else "否",
                        ]
                    )

                preview_text += "(详细匹配情况将在生成时显示)"
                return preview_text, table_data

            except Exception as exc:
                return f"解析乐谱时出错: {str(exc)}", []

        score_upload.change(
            fn=preview_score, inputs=[score_upload], outputs=[score_info, notes_match_table]
        )


def build_advanced_ui():
    with gr.Blocks(title="高级音频处理与音乐制作系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
        # 🎛️ 高级音频处理与音乐制作系统

        ## 功能介绍
        1. **音频识别**：自动检测音频频率并转换为音名
        2. **音频处理**：支持变速、变调、淡入淡出
        3. **音乐制作**：根据乐谱或手动编排制作音乐
        4. **频谱分析**：可视化音频特征
        """
        )

        with gr.Tabs():
            with gr.TabItem("🎙️ 音频上传与识别"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(label="上传音频文件", type="filepath")
                        target_note = gr.Textbox(
                            label="目标音高（可选）",
                            placeholder="例如：C4, D#4, Gb5",
                            value="",
                        )

                        with gr.Row():
                            auto_detect = gr.Checkbox(label="自动检测音高", value=True)
                            analysis_mode = gr.Radio(
                                choices=["simple", "enhanced"],
                                label="分析模式",
                                value="simple",
                                info="简单模式：频谱图+波形图 | 增强模式：多种分析图表",
                            )

                        btn_analyze = gr.Button("分析音频", variant="primary")

                    with gr.Column(scale=2):
                        result_text = gr.Markdown(label="分析结果")
                        clip_id_output = gr.Number(label="片段ID", visible=False)
                        spectrogram = gr.Plot(label="频谱图分析")
                        enhanced_analysis = gr.Plot(label="增强分析", visible=False)

                        def toggle_analysis(analysis_mode):
                            if analysis_mode == "enhanced":
                                return gr.Plot(visible=True)
                            return gr.Plot(visible=False)

                        analysis_mode.change(
                            fn=toggle_analysis, inputs=[analysis_mode], outputs=[enhanced_analysis]
                        )

                btn_analyze.click(
                    fn=handle_audio_upload,
                    inputs=[audio_input, target_note, auto_detect, analysis_mode],
                    outputs=[result_text, clip_id_output, spectrogram, enhanced_analysis],
                )

                # 频谱图说明
                with gr.Accordion("📈 频谱图解读指南", open=False):
                    gr.Markdown(
                        """
                    ### 如何读懂频谱图：

                    1. **时间轴（X轴）**：从左到右表示音频的时间进度
                    2. **频率轴（Y轴）**：从下到上表示声音频率（低音在下，高音在上）
                    3. **颜色深浅**：表示音量大小
                       - **深色/蓝色**：安静的声音
                       - **亮色/黄色**：响亮的声音
                    4. **红色虚线**：检测到的主音高频率
                    5. **底部波形图**：音频的振幅变化

                    ### 常见音频在频谱图上的表现：
                    - **纯音/乐器单音**：一条清晰的水平线
                    - **人声/复杂音色**：多条水平线（基频+泛音）
                    - **噪音/打击乐**：垂直的色块（短暂爆发）
                    - **静音**：深色或黑色区域

                    ### 如何判断音高：
                    - 寻找最亮的水平线条
                    - 对照右侧频率标尺
                    - 红色虚线标记的是系统检测到的主频率
                    """
                    )

            with gr.TabItem("🛠️ 音频处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        clip_id_input = gr.Number(label="片段ID", value=0, precision=0)
                        operation = gr.Radio(
                            choices=["time_stretch", "pitch_shift"],
                            label="处理类型",
                            value="time_stretch",
                        )
                        value_input = gr.Slider(
                            label="参数值",
                            minimum=0.1,
                            maximum=5.0,
                            value=1.0,
                            step=0.1,
                            visible=True,
                        )

                        def update_slider(operation):
                            if operation == "time_stretch":
                                return gr.Slider(
                                    minimum=0.1,
                                    maximum=5.0,
                                    value=1.0,
                                    step=0.1,
                                    label="目标时长（秒）",
                                )
                            return gr.Slider(
                                minimum=-12,
                                maximum=12,
                                value=0,
                                step=0.5,
                                label="半音移动",
                            )

                        operation.change(fn=update_slider, inputs=[operation], outputs=[value_input])

                        btn_process = gr.Button("处理音频", variant="primary")
                        process_result = gr.Markdown(label="处理结果")

                    with gr.Column(scale=2):
                        audio_preview = gr.Audio(label="处理结果预览", type="numpy")

                btn_process.click(
                    fn=process_audio_clip,
                    inputs=[clip_id_input, operation, value_input],
                    outputs=[process_result, audio_preview],
                )

            with gr.TabItem("🎵 音乐制作"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            """
                        ### 音乐编排说明

                        格式：`拍数:片段ID, 拍数:片段ID, ...`

                        示例：
                        ```
                        0:0, 1:1, 2:2, 4:3
                        ```

                        表示：
                        - 第0拍使用片段0
                        - 第1拍使用片段1  
                        - 第2拍使用片段2
                        - 第4拍使用片段3
                        """
                        )

                        clip_assignments = gr.Textbox(
                            label="片段分配",
                            placeholder="格式: 拍数:片段ID, 拍数:片段ID,...",
                            lines=3,
                        )
                        tempo_input = gr.Slider(
                            label="速度 (BPM)",
                            minimum=40,
                            maximum=240,
                            value=120,
                            step=10,
                        )
                        btn_compose = gr.Button("生成音乐", variant="primary")
                        compose_result = gr.Markdown(label="生成结果")

                    with gr.Column(scale=2):
                        composition_audio = gr.Audio(label="生成音乐", type="numpy")

                btn_compose.click(
                    fn=generate_music_from_clips,
                    inputs=[clip_assignments, tempo_input],
                    outputs=[compose_result, composition_audio],
                )

            with gr.TabItem("📁 片段管理"):
                clip_manager = require_clip_manager()

                def update_clips_table():
                    clips = clip_manager.get_all_clips()
                    table_data = []
                    for clip in clips:
                        note_info = clip.get("note_info", {})
                        table_data.append(
                            [
                                clip["id"],
                                clip["filename"],
                                note_info.get("note", "未知"),
                                f"{note_info.get('frequency', 0):.1f}"
                                if note_info.get("frequency")
                                else "未知",
                                f"{note_info.get('cents', 0):+.1f}"
                                if note_info.get("cents") is not None
                                else "",
                                f"{clip['duration']:.2f}",
                                clip["created_at"],
                            ]
                        )
                    return table_data

                with gr.Row():
                    clips_table = gr.Dataframe(
                        headers=["ID", "文件名", "音名", "频率", "偏差", "时长", "创建时间"],
                        label="所有音频片段",
                        datatype=["number", "str", "str", "str", "str", "str", "str"],
                        row_count=10,
                        col_count=7,
                        interactive=False,
                    )

                with gr.Row():
                    btn_refresh = gr.Button("刷新列表")
                    delete_clip_id = gr.Number(label="删除片段ID", value=0, precision=0)
                    btn_delete = gr.Button("删除片段", variant="stop")

                with gr.Row():
                    btn_cleanup = gr.Button("清理孤立文件", variant="secondary")

                def cleanup_orphaned():
                    clip_manager.cleanup_orphaned_files()
                    return "已清理孤立文件", update_clips_table()

                btn_cleanup.click(fn=cleanup_orphaned, inputs=[], outputs=[compose_result, clips_table])

                def delete_selected_clip(clip_id):
                    success = clip_manager.delete_clip(int(clip_id))
                    if success:
                        return f"✅ 已删除片段 {clip_id}", update_clips_table()
                    return f"❌ 删除失败，片段 {clip_id} 不存在", update_clips_table()

                btn_refresh.click(fn=update_clips_table, inputs=[], outputs=[clips_table])

                btn_delete.click(
                    fn=delete_selected_clip, inputs=[delete_clip_id], outputs=[compose_result, clips_table]
                )

            build_music_composition_tab()
        gr.Markdown(
            """
        ## 📘 使用说明

        ### 1. 音频识别
        - 上传音频文件（支持wav, mp3等格式）
        - 系统会自动检测音高并显示频谱图
        - 可输入目标音高进行比较

        ### 2. 音频处理
        - **时间拉伸**：调整音频时长而不改变音高
        - **音高移动**：调整音高而不改变时长
        - 处理后的音频会保存为新片段

        ### 3. 音乐制作
        - 将音频片段分配到特定的拍数位置
        - 调整音乐速度（BPM）
        - 系统会自动拼接生成完整音乐

        ### 4. 频谱图解读
        - **水平线**：稳定的音高
        - **垂直线**：瞬时声音（如鼓点）
        - **颜色深浅**：音量大小
        - **红色虚线**：检测到的主频率

        ## ⚙️ 安装说明
        ```bash
        # 基本依赖
        pip install gradio librosa numpy soundfile matplotlib scipy

        # 解决中文字体问题（Windows）
        # 确保系统已安装中文字体（如微软雅黑）

        # 解决中文字体问题（Linux）
        sudo apt-get install fonts-wqy-microhei
        ```
        """
        )

    return demo
