import math
import os
import re
import time
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal

# ========= 解决中文字体问题 =========
import matplotlib
try:
    # 尝试设置中文字体
    font_list = []
    
    # Windows字体路径
    if os.name == 'nt':
        font_dirs = [
            'C:/Windows/Fonts',  # Windows系统字体
            os.path.expanduser('~/.fonts'),  # 用户字体
        ]
        # 常见中文字体名称
        chinese_fonts = [
            'msyh.ttc',  # 微软雅黑
            'simhei.ttf',  # 黑体
            'simsun.ttc',  # 宋体
            'simkai.ttf',  # 楷体
            'STHeiti Light.ttc',  # 华文黑体（Mac）
            'PingFang.ttc',  # 苹方（Mac）
        ]
    else:
        # Linux/Mac字体路径
        font_dirs = [
            '/usr/share/fonts',
            '/usr/local/share/fonts',
            os.path.expanduser('~/.fonts'),
            os.path.expanduser('~/Library/Fonts'),  # Mac
        ]
        chinese_fonts = [
            'wqy-microhei.ttc',  # 文泉驿微米黑
            'NotoSansCJK-Regular.ttc',  # Noto字体
            'SourceHanSansSC-Regular.otf',  # 思源黑体
        ]
    
    # 查找可用的中文字体
    available_fonts = []
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            for font_file in chinese_fonts:
                font_path = os.path.join(font_dir, font_file)
                if os.path.exists(font_path):
                    available_fonts.append(font_path)
                    print(f"找到中文字体: {font_path}")
    
    # 如果有找到中文字体，使用第一个
    if available_fonts:
        matplotlib.font_manager.fontManager.addfont(available_fonts[0])
        font_name = matplotlib.font_manager.FontProperties(fname=available_fonts[0]).get_name()
        matplotlib.rcParams['font.sans-serif'] = [font_name]
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print(f"已设置中文字体: {font_name}")
    else:
        print("未找到中文字体，将使用默认字体")
        
except Exception as e:
    print(f"设置字体时出错: {e}")
    # 使用默认字体，但尝试设置支持中文的字体
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False

# ========= 配置管理 =========

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
    clip_dir: str = "clips"
    output_dir: str = "output"
    config_file: str = "config.json"

config = AppConfig()
os.makedirs(config.clip_dir, exist_ok=True)
os.makedirs(config.output_dir, exist_ok=True)

# ========= 音频处理工具函数 =========

_clip_index_cache = None  # 缓存索引，避免每次重建

def build_clip_index():
    """
    构建音频片段的音高索引，加速查找。
    返回结构: {rounded_midi: [(clip_info, original_midi, confidence), ...]}
    """
    global _clip_index_cache
    if _clip_index_cache is not None:
        return _clip_index_cache
    
    index = {}
    available_clips = clip_manager.get_all_clips()
    
    for clip in available_clips:
        note_info = clip.get('note_info', {})
        if note_info and note_info.get('frequency') and note_info.get('confidence'):
            clip_freq = note_info['frequency']
            clip_midi = freq_to_midi(clip_freq)  # 精确的浮点数MIDI
            confidence = note_info.get('confidence', 0.5)
            
            # 将MIDI音高四舍五入到最接近的整数，作为索引键
            rounded_midi = int(round(clip_midi))
            
            if rounded_midi not in index:
                index[rounded_midi] = []
            
            index[rounded_midi].append({
                'clip': clip,
                'exact_midi': clip_midi,     # 保存精确值用于计算
                'confidence': confidence,
                # 可以在这里扩展存储 instrument_tag 等元数据
            })
    
    _clip_index_cache = index
    print(f"[索引构建完成] 共 {len(available_clips)} 个片段，索引到 {len(index)} 个不同音高。")
    return index

# ========= 乐理工具函数 =========

def note_to_midi(note: str) -> Optional[int]:
    """音名转MIDI编号，支持扩展格式"""
    if not note:
        return None
    
    note = note.strip().upper()
    
    # 匹配格式：音名[升降号]八度（例如：C4, D#4, Gb3, A♯5）
    pattern = r'([A-G])([#♯b♭]?)(-?\d+)'
    match = re.match(pattern, note)
    
    if not match:
        return None
    
    note_name, accidental, octave_str = match.groups()
    
    # 基本音名映射
    base_notes = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    
    if note_name not in base_notes:
        return None
    
    midi_base = base_notes[note_name]
    
    # 处理升降号
    if accidental in ('#', '♯'):
        midi_base += 1
    elif accidental in ('b', '♭'):
        midi_base -= 1
    
    try:
        octave = int(octave_str)
        midi_number = (octave + 1) * 12 + midi_base
        return midi_number
    except ValueError:
        return None

def midi_to_note(midi: int) -> str:
    """MIDI编号转音名"""
    if not 0 <= midi <= 127:
        return ""
    
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = midi // 12 - 1
    note_name = notes[midi % 12]
    
    # 将升号替换为更易读的格式
    if '#' in note_name:
        base_note = note_name[0]
        return f"{base_note}♯{octave}"
    return f"{note_name}{octave}"

def midi_to_freq(midi: int) -> float:
    """MIDI转频率"""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))

def freq_to_midi(freq: float) -> float:
    """频率转MIDI（浮点数，更精确）"""
    return 69 + 12 * math.log2(freq / 440.0)

def freq_to_note(freq: float) -> Tuple[str, float]:
    """频率转音名和音分偏差"""
    midi_float = freq_to_midi(freq)
    midi_int = round(midi_float)
    
    # 计算音分偏差
    cents = (midi_float - midi_int) * 100
    
    note_name = midi_to_note(midi_int)
    return note_name, cents

def detect_pitch_advanced(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    高级音高检测，返回详细信息
    """
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    
    # 裁剪静音
    y_trimmed, _ = librosa.effects.trim(
        y, 
        top_db=config.silence_threshold_db,
        frame_length=2048,
        hop_length=512
    )
    
    if len(y_trimmed) < sr * config.min_clip_duration:
        return {
            'frequency': None,
            'note': None,
            'cents': 0,
            'confidence': 0,
            'stable': False
        }
    
    # 使用多种方法检测音高
    freqs = []
    confidences = []
    
    # 方法1: YIN算法
    f0_yin = librosa.yin(
        y_trimmed,
        fmin=config.min_freq,
        fmax=config.max_freq,
        sr=sr,
        frame_length=2048,
        hop_length=512
    )
    f0_yin = f0_yin[f0_yin > 0]
    if len(f0_yin) > 0:
        freqs.append(np.median(f0_yin))
        # 用稳定度作为置信度
        confidences.append(1.0 - (np.std(f0_yin) / np.mean(f0_yin)) if np.mean(f0_yin) > 0 else 0)
    
    # 方法2: PYIN算法（更准确但更慢）
    try:
        f0_pyin, voiced_flag, voiced_probs = librosa.pyin(
            y_trimmed,
            fmin=config.min_freq,
            fmax=config.max_freq,
            sr=sr
        )
        f0_pyin = f0_pyin[voiced_flag]
        if len(f0_pyin) > 0:
            freqs.append(np.median(f0_pyin))
            confidences.append(np.mean(voiced_probs[voiced_flag]))
    except:
        pass
    
    # 方法3: 谐波乘积谱（对音乐信号更准确）
    try:
        f0_hps = _detect_pitch_hps(y_trimmed, sr)
        if f0_hps:
            freqs.append(f0_hps)
            confidences.append(0.7)
    except:
        pass
    
    if not freqs:
        return {
            'frequency': None,
            'note': None,
            'cents': 0,
            'confidence': 0,
            'stable': False
        }
    
    # 加权平均
    weights = np.array(confidences)
    if weights.sum() == 0:
        weights = np.ones(len(freqs))
    
    weighted_freq = np.average(freqs, weights=weights)
    avg_confidence = np.mean(confidences)
    
    # 转换为音名
    note_name, cents = freq_to_note(weighted_freq)
    
    # 判断是否稳定
    is_stable = avg_confidence > config.confidence_threshold and len(y_trimmed) > sr * 0.2
    
    return {
        'frequency': float(weighted_freq) if weighted_freq is not None else None,
        'note': note_name,
        'cents': float(cents) if cents is not None else None,
        'confidence': float(avg_confidence),
        'stable': bool(is_stable),
        'duration': float(len(y_trimmed) / sr)
    }

def _detect_pitch_hps(y: np.ndarray, sr: int) -> Optional[float]:
    """谐波乘积谱音高检测"""
    n_fft = 2048
    hop_length = 512
    
    # 计算频谱
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    
    # 谐波乘积谱
    hps = S.copy()
    for harmonic in (2, 3, 4):
        downsampled = signal.resample_poly(
            S, 
            1, 
            harmonic, 
            axis=0
        )
        hps = hps[:downsampled.shape[0]] * downsampled
    
    # 寻找峰值
    hps_mean = np.mean(hps, axis=1)
    peaks, properties = signal.find_peaks(hps_mean, height=np.max(hps_mean)*0.1)
    
    if len(peaks) == 0:
        return None
    
    # 找到最低频率的峰值（基频）
    min_peak_idx = peaks[np.argmin(peaks)]
    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)[min_peak_idx]
    
    # 限制在合理范围内
    if config.min_freq <= freq <= config.max_freq:
        return freq
    return None

# ========= 音频处理函数 =========

def time_stretch(y: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
    """
    时间拉伸（变速不变调）
    """
    current_duration = len(y) / sr
    rate = current_duration / target_duration
    
    # 限制拉伸范围
    rate = np.clip(rate, *config.time_stretch_range)
    
    # 使用librosa的时间拉伸
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    
    # 确保长度准确
    target_samples = int(target_duration * sr)
    if len(y_stretched) > target_samples:
        y_stretched = y_stretched[:target_samples]
    else:
        y_stretched = np.pad(
            y_stretched, 
            (0, target_samples - len(y_stretched)), 
            mode='constant'
        )
    
    return y_stretched

def pitch_shift(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """
    音高平移（变调不变速）
    """
    # 限制移动范围
    semitones = np.clip(semitones, *config.pitch_shift_range)
    
    return librosa.effects.pitch_shift(
        y, 
        sr=sr, 
        n_steps=semitones,
        bins_per_octave=12
    )

def normalize_audio(y: np.ndarray) -> np.ndarray:
    """音频归一化"""
    if len(y) == 0:
        return y
    
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        return y / max_amp * 0.9  # 留一点headroom
    return y

def apply_fade(y: np.ndarray, sr: int, fade_in: float = 0.01, fade_out: float = 0.01) -> np.ndarray:
    """应用淡入淡出"""
    if len(y) == 0:
        return y
    
    fade_in_samples = int(fade_in * sr)
    fade_out_samples = int(fade_out * sr)
    
    # 创建淡入淡出窗口
    if fade_in_samples > 0:
        fade_in_window = np.linspace(0, 1, fade_in_samples)
        if fade_in_samples <= len(y):
            y[:fade_in_samples] *= fade_in_window
    
    if fade_out_samples > 0:
        fade_out_window = np.linspace(1, 0, fade_out_samples)
        if fade_out_samples <= len(y):
            y[-fade_out_samples:] *= fade_out_window
    
    return y

# ========= 频谱图可视化 =========

def create_spectrogram(y: np.ndarray, sr: int, detected_freq: float = None) -> plt.Figure:
    """
    创建更易读的频谱图，包含中文标签和详细说明
    
    频谱图解释：
    - X轴：时间（秒）
    - Y轴：频率（赫兹Hz），对数坐标显示（低音在下，高音在上）
    - 颜色：音量强度（深色=安静，亮色=响亮）
    - 水平线：检测到的基频
    """
    plt.figure(figsize=(12, 8))
    
    # 计算频谱图
    n_fft = 2048
    hop_length = 512
    
    # 使用mel频谱图，更符合人耳听觉
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # 显示频谱图
    ax1 = plt.subplot(2, 1, 1)
    img = librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, 
                                   x_axis='time', y_axis='mel',
                                   cmap='viridis')
    
    plt.colorbar(img, format='%+2.0f dB', ax=ax1)
    plt.title('音频频谱图 (Mel Spectrogram)', fontsize=14, fontweight='bold')
    plt.xlabel('时间 (秒)')
    plt.ylabel('频率 (Hz) - Mel刻度')
    
    # 标记检测到的频率
    if detected_freq:
        # 将频率转换为mel刻度
        mel_freq = librosa.hz_to_mel(detected_freq)
        plt.axhline(y=mel_freq, color='red', linestyle='--', linewidth=2, 
                   alpha=0.8, label=f'检测基频: {detected_freq:.1f} Hz')
        
        # 在右侧显示频率值
        plt.text(plt.xlim()[1] * 1.02, mel_freq, 
                f'{detected_freq:.0f} Hz', 
                color='red', va='center', fontsize=10)
        
        plt.legend(loc='upper right')
    
    # 添加网格，提高可读性
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 在下方显示波形图
    ax2 = plt.subplot(2, 1, 2)
    time = np.linspace(0, len(y)/sr, len(y))
    plt.plot(time, y, color='blue', alpha=0.7, linewidth=0.5)
    plt.fill_between(time, y, 0, alpha=0.3, color='blue')
    
    plt.title('音频波形', fontsize=14, fontweight='bold')
    plt.xlabel('时间 (秒)')
    plt.ylabel('振幅')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 设置x轴范围一致
    ax1.set_xlim([0, len(y)/sr])
    ax2.set_xlim([0, len(y)/sr])
    
    plt.tight_layout()
    
    return plt.gcf()

def create_enhanced_analysis(y: np.ndarray, sr: int, detected_info: Dict) -> plt.Figure:
    """
    创建增强分析图，包含多种可视化
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 频谱图
    ax1 = plt.subplot(3, 2, 1)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('频谱图')
    
    # 2. 波形图
    ax2 = plt.subplot(3, 2, 2)
    time = np.linspace(0, len(y)/sr, len(y))
    plt.plot(time, y, color='blue', alpha=0.7, linewidth=0.5)
    plt.fill_between(time, y, 0, alpha=0.3, color='blue')
    plt.title('波形图')
    plt.xlabel('时间 (秒)')
    plt.ylabel('振幅')
    plt.grid(True, alpha=0.3)
    
    # 3. 频谱图（线性频率）
    ax3 = plt.subplot(3, 2, 3)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('频谱图（对数频率）')
    
    # 4. 基频轨迹（如果有）
    if detected_info.get('frequency'):
        ax4 = plt.subplot(3, 2, 4)
        
        # 计算基频轨迹
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        times = librosa.times_like(f0, sr=sr)
        
        plt.plot(times, f0, label='基频轨迹', color='red', linewidth=2)
        plt.axhline(y=detected_info['frequency'], color='green', linestyle='--', 
                   label=f"检测频率: {detected_info['frequency']:.1f} Hz")
        plt.title('基频轨迹')
        plt.xlabel('时间 (秒)')
        plt.ylabel('频率 (Hz)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 5. 频谱质心
    ax5 = plt.subplot(3, 2, 5)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    times = librosa.times_like(spectral_centroids, sr=sr)
    plt.plot(times, spectral_centroids, color='purple')
    plt.title('频谱质心')
    plt.xlabel('时间 (秒)')
    plt.ylabel('频率 (Hz)')
    plt.grid(True, alpha=0.3)
    
    # 6. 过零率
    ax6 = plt.subplot(3, 2, 6)
    zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
    times = librosa.times_like(zero_crossings, sr=sr)
    plt.plot(times, zero_crossings, color='orange')
    plt.title('过零率')
    plt.xlabel('时间 (秒)')
    plt.ylabel('过零率')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ========= 音频文件管理 =========

def convert_to_serializable(obj):
    """将numpy类型转换为Python原生类型以便JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

class AudioClipManager:
    """音频片段管理器"""
    
    def __init__(self):
        self.clips = []
        self.load_clips()
    
    def load_clips(self):
        """加载已保存的片段"""
        if os.path.exists('clips.json'):
            with open('clips.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.clips = data.get('clips', [])
    
    def save_clips(self):
        """保存片段信息"""
        # 转换所有数据为可序列化的Python原生类型
        clips_serializable = convert_to_serializable(self.clips)
        data = {'clips': clips_serializable}
        with open('clips.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add_clip(self, audio_data: np.ndarray, sr: int, 
                 note_info: Dict = None, metadata: Dict = None) -> str:
        """添加音频片段"""
        # 生成文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"clip_{timestamp}.wav"
        filepath = os.path.join(config.clip_dir, filename)
        
        # 保存音频文件
        sf.write(filepath, audio_data, sr)
        
        # 如果没有提供音高信息，则检测
        if note_info is None:
            note_info = detect_pitch_advanced(audio_data, sr)
        
        # 创建片段信息
        clip_info = {
            'id': int(len(self.clips)),
            'filename': filename,
            'filepath': filepath,
            'sample_rate': int(sr),
            'duration': float(len(audio_data) / sr),
            'note_info': convert_to_serializable(note_info) if note_info else {},
            'metadata': convert_to_serializable(metadata) if metadata else {},
            'created_at': timestamp,
            'tags': []
        }
        
        self.clips.append(clip_info)
        self.save_clips()
        clear_clip_index_cache()
        return clip_info
    
    def get_clip_by_note(self, target_note: str, tolerance_cents: float = 50) -> List[Dict]:
        """根据音名获取片段"""
        target_midi = note_to_midi(target_note)
        if target_midi is None:
            return []
        
        matching_clips = []
        for clip in self.clips:
            note_info = clip.get('note_info', {})
            if note_info.get('note') and note_info.get('frequency'):
                clip_midi = freq_to_midi(note_info['frequency'])
                cents_diff = (clip_midi - target_midi) * 100
                
                if abs(cents_diff) <= tolerance_cents:
                    matching_clips.append({
                        **clip,
                        'cents_diff': cents_diff
                    })
        
        # 按偏差排序
        matching_clips.sort(key=lambda x: abs(x['cents_diff']))
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
                if os.path.exists(clip['filepath']):
                    os.remove(clip['filepath'])
                    print(f"已删除音频文件: {clip['filepath']}")
            except Exception as e:
                print(f"删除音频文件时出错 {clip['filepath']}: {e}")
            
            # 更新ID
            for i, c in enumerate(self.clips):
                c['id'] = i
            
            self.save_clips()
            clear_clip_index_cache()
            return True
        return False
    
    def cleanup_orphaned_files(self):
        """清理没有对应记录的音频文件"""
        # 获取所有应该存在的文件路径
        expected_files = {clip['filepath'] for clip in self.clips}
        
        # 遍历clips目录
        clips_dir = config.clip_dir
        for filename in os.listdir(clips_dir):
            if filename.endswith('.wav'):
                filepath = os.path.join(clips_dir, filename)
                if filepath not in expected_files:
                    try:
                        os.remove(filepath)
                        print(f"清理孤立文件: {filename}")
                    except Exception as e:
                        print(f"清理文件失败 {filename}: {e}")

    def delete_all_clips(self):
        """删除所有片段和对应的文件"""
        deleted_files = []
        for clip in self.clips:
            try:
                if os.path.exists(clip['filepath']):
                    os.remove(clip['filepath'])
                    deleted_files.append(clip['filename'])
            except Exception as e:
                print(f"删除文件失败 {clip['filename']}: {e}")
        
        self.clips = []
        self.save_clips()
        clear_clip_index_cache()
        return deleted_files

# ========= Gradio界面函数 =========

clip_manager = AudioClipManager()

def handle_audio_upload(audio_input, target_note, auto_detect, analysis_mode):
    """处理音频上传"""
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
    
    if note_info['frequency']:
        message.append(f"检测到频率: **{note_info['frequency']:.1f} Hz**")
        message.append(f"音名: **{note_info['note']}**")
        message.append(f"音分偏差: **{note_info['cents']:+.1f} cents**")
        message.append(f"置信度: **{note_info['confidence']:.2%}**")
        
        if note_info['stable']:
            message.append("✅ 音高稳定")
        else:
            message.append("⚠ 音高不稳定，可能包含滑音或多音")
    else:
        message.append("⚠ 无法检测到稳定音高")
    
    # 如果有目标音高，进行比较
    if target_note:
        target_midi = note_to_midi(target_note)
        if target_midi:
            target_freq = midi_to_freq(target_midi)
            if note_info['frequency']:
                cents_diff = (freq_to_midi(note_info['frequency']) - target_midi) * 100
                message.append(f"目标音高: **{target_note}** ({target_freq:.1f} Hz)")
                message.append(f"偏差: **{cents_diff:+.1f} cents**")
                
                if abs(cents_diff) <= 50:
                    message.append("✅ 在可接受范围内 (±50 cents)")
                else:
                    message.append("⚠ 偏差较大")
        else:
            message.append(f"⚠ 目标音高 '{target_note}' 格式错误")
    
    # 保存片段
    clip_info = clip_manager.add_clip(
        y, sr, 
        note_info=convert_to_serializable(note_info) if note_info else None,
        metadata={
            'target_note': str(target_note) if target_note else "",
            'upload_time': str(time.strftime("%Y-%m-%d %H:%M:%S"))
        }
    )
    
    # 生成图表
    if analysis_mode == "simple":
        fig = create_spectrogram(y, sr, note_info.get('frequency'))
        fig2 = None
    else:
        fig = create_spectrogram(y, sr, note_info.get('frequency'))
        fig2 = create_enhanced_analysis(y, sr, note_info)
    
    return "\n".join(message), clip_info['id'], fig, fig2

def process_audio_clip(clip_id, operation, value):
    """处理音频片段（变速/变调）"""
    if not 0 <= clip_id < len(clip_manager.clips):
        return "无效的片段ID", None
    
    clip = clip_manager.clips[clip_id]
    y, sr = sf.read(clip['filepath'])
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    
    if operation == "time_stretch":
        target_duration = float(value)
        y_processed = time_stretch(y, sr, target_duration)
        message = f"时长调整为 {target_duration:.2f} 秒"
    elif operation == "pitch_shift":
        semitones = float(value)
        y_processed = pitch_shift(y, sr, semitones)
        message = f"音高调整 {semitones:+.1f} 个半音"
    else:
        return "未知操作", None
    
    # 应用淡入淡出
    y_processed = apply_fade(y_processed, sr)
    
    # 保存处理后的音频
    processed_info = clip_manager.add_clip(
        y_processed, sr,
        metadata={
            'original_clip_id': clip_id,
            'operation': operation,
            'value': value,
            'processed_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }
    )
    
    return f"✅ {message} (新片段ID: {processed_info['id']})", (sr, y_processed)

def generate_music_from_clips(clip_assignments, tempo):
    """从片段生成音乐"""
    sr = config.sample_rate
    beat_duration = 60.0 / tempo
    
    # 解析片段分配（格式: "时间拍:片段ID,时间拍:片段ID,..."）
    assignments = []
    for assignment in clip_assignments.split(','):
        if ':' in assignment:
            beat_str, clip_id_str = assignment.split(':')
            try:
                beat = float(beat_str.strip())
                clip_id = int(clip_id_str.strip())
                assignments.append((beat, clip_id))
            except:
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
            y, _ = sf.read(clip['filepath'])
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            
            # 调整到标准时长（1拍）
            target_samples = int(beat_duration * sr)
            if len(y) > target_samples:
                y = y[:target_samples]
            else:
                y = np.pad(y, (0, target_samples - len(y)), mode='constant')
            
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

# ========= 音乐生成界面 ===========
def build_music_composition_tab():
    """构建全新的自动音乐制作界面"""
    
    with gr.TabItem("🎹 智能音乐制作"):
        gr.Markdown("""
        ## 🎼 智能音乐制作工作台
        上传乐谱 → 自动匹配音频片段 → 智能变调处理 → 生成完整音乐
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 乐谱上传区域
                gr.Markdown("### 1. 上传乐谱")
                score_upload = gr.File(
                    label="选择乐谱文件",
                    file_types=[".xml", ".musicxml", ".mid", ".midi"],
                    type="filepath"
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
                        info="值越小匹配要求越严格"
                    )
                    
                    use_pitch_shift = gr.Checkbox(
                        label="启用智能变调",
                        value=True,
                        info="对不匹配的音符自动变调处理"
                    )
                
                tempo_input = gr.Slider(
                    label="演奏速度 (BPM)",
                    minimum=40,
                    maximum=240,
                    value=120,
                    step=5
                )
                
                # 生成按钮
                btn_generate = gr.Button("🎵 自动生成音乐", variant="primary", size="lg")
                generation_status = gr.Markdown("准备生成...", label="生成状态")
                
            with gr.Column(scale=2):
                # 生成结果区域
                gr.Markdown("### 3. 生成结果")
                
                with gr.Tabs():
                    with gr.TabItem("🎧 试听音乐"):
                        composition_audio = gr.Audio(
                            label="生成音乐",
                            type="numpy"
                        )
                    
                    with gr.TabItem("📊 生成报告"):
                        generation_report = gr.Markdown(
                            "生成报告将在此显示...",
                            label="详细报告"
                        )
                    
                    with gr.TabItem("🎵 音符匹配详情"):
                        notes_match_table = gr.Dataframe(
                            headers=["序号", "音名", "匹配片段", "变调(半音)", "状态", "音轨", "乐器"],
                            label="音符匹配情况",
                            datatype=["str", "str", "str", "str", "str", "str", "str"],
                            row_count=10,
                            interactive=False
                        )
        
        # 连接生成按钮
        btn_generate.click(
            fn=auto_generate_music_from_score,
            inputs=[score_upload, tempo_input, match_tolerance, use_pitch_shift],
            outputs=[composition_audio, generation_report, notes_match_table, generation_status]
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
                preview_text = f"### 乐谱解析成功！\n"
                preview_text += f"**音符总数**: {len(notes)}\n"
                preview_text += f"**音高范围**: {notes[0]['note_name']} 到 {notes[-1]['note_name']}\n"
                preview_text += f"**总时长**: {sum(n['duration'] for n in notes):.2f} 拍\n\n"
                preview_text += "**前10个音符**:\n"
                
                # 构建表格数据
                table_data = []
                for i, note in enumerate(notes[:10]):
                    table_data.append([
                        i+1,
                        note['note_name'],
                        f"{note['duration']:.2f}拍",
                        f"{note['start_time']:.2f}拍",
                        "是" if note['matched'] else "否"
                    ])
                
                preview_text += "(详细匹配情况将在生成时显示)"
                return preview_text, table_data
                
            except Exception as e:
                return f"解析乐谱时出错: {str(e)}", []
        
        score_upload.change(
            fn=preview_score,
            inputs=[score_upload],
            outputs=[score_info, notes_match_table]
        )

# ========= 核心音乐生成函数 =========

# ========= 核心音乐生成函数 =========

def parse_score_notes(filepath: str) -> List[Dict]:
    """
    专业解析乐谱文件，提取音符信息
    支持 MusicXML 和 MIDI 格式，MIDI解析现在支持完整音符追踪、多音轨、休止符检测
    """
    notes = []
    
    if not filepath or not os.path.exists(filepath):
        print(f"文件不存在或路径错误: {filepath}")
        return notes
    
    file_ext = os.path.splitext(filepath)[1].lower()
    
    try:
        if file_ext in ['.xml', '.musicxml']:
            # ============ MusicXML 解析部分 ============
            # 尝试使用 partitura 解析（更专业）
            try:
                import partitura as pt
                score = pt.load_score(filepath)
                print(f"使用 partitura 解析 XML，共找到 {len(score.notes)} 个音符")
                
                for i, note in enumerate(score.notes):
                    notes.append({
                        'midi_pitch': int(note.midi_pitch),
                        'note_name': note.step + str(note.octave),
                        'duration': float(note.duration),
                        'start_time': float(note.start),
                        'velocity': int(note.velocity) if hasattr(note, 'velocity') else 64,
                        'matched': False,
                        'clip_id': None,
                        'pitch_shift': 0,
                        'track': 0,  # XML通常不分轨
                        'instrument': 'piano',  # 默认
                        'source': 'xml'
                    })
                    
            except ImportError:
                print("未安装 partitura，使用 music21 解析 XML")
                # 回退到 music21
                import music21 as m21
                score = m21.converter.parse(filepath)
                
                # 获取所有音符
                all_notes = list(score.flat.notesAndRests)
                print(f"使用 music21 解析 XML，共找到 {len(all_notes)} 个音符/休止符")
                
                for element in all_notes:
                    if isinstance(element, m21.note.Note):
                        notes.append({
                            'midi_pitch': element.pitch.midi,
                            'note_name': str(element.pitch),
                            'duration': float(element.duration.quarterLength),
                            'start_time': float(element.offset),
                            'velocity': 64,
                            'matched': False,
                            'clip_id': None,
                            'pitch_shift': 0,
                            'track': 0,
                            'instrument': 'piano',
                            'source': 'xml'
                        })
                    elif isinstance(element, m21.note.Rest):
                        # 将休止符记录为特殊音符，midi_pitch为-1
                        notes.append({
                            'midi_pitch': -1,  # 休止符标识
                            'note_name': 'REST',
                            'duration': float(element.duration.quarterLength),
                            'start_time': float(element.offset),
                            'velocity': 0,
                            'matched': False,
                            'clip_id': None,
                            'pitch_shift': 0,
                            'track': 0,
                            'instrument': 'rest',
                            'source': 'xml'
                        })
        
        elif file_ext in ['.mid', '.midi']:
            # ============ 专业MIDI解析部分 ============
            import mido
            
            print(f"开始解析 MIDI 文件: {os.path.basename(filepath)}")
            midi = mido.MidiFile(filepath)
            
            # 获取MIDI文件的基本信息
            ticks_per_beat = midi.ticks_per_beat
            print(f"MIDI基本信息 - 音轨数: {len(midi.tracks)}, 每拍Tick数: {ticks_per_beat}, 类型: {midi.type}")
            
            # 存储各音轨的当前时间和活动音符
            track_info = []
            for i in range(len(midi.tracks)):
                track_info.append({
                    'current_time': 0,  # 当前绝对时间（tick）
                    'active_notes': {},  # 正在播放的音符: {note_number: start_tick}
                    'tempo': 500000,  # 默认tempo (120 BPM)
                    'time_signature': (4, 4),  # 默认拍号
                    'key_signature': 'C',  # 默认调号
                    'program': 0,  # 默认乐器 (Acoustic Grand Piano)
                })
            
            # 第一遍：收集所有音符开始和结束事件
            note_events = []  # (absolute_tick, track_index, note_number, velocity, event_type)
            
            for track_idx, track in enumerate(midi.tracks):
                current_tick = 0
                
                print(f"  解析音轨 {track_idx}: {track.name if track.name else '未命名'}, 消息数: {len(track)}")
                
                for msg in track:
                    current_tick += msg.time
                    
                    if msg.type == 'note_on':
                        if msg.velocity > 0:
                            # 音符开始
                            note_events.append((current_tick, track_idx, msg.note, msg.velocity, 'start'))
                        else:
                            # velocity=0 的 note_on 等价于 note_off
                            note_events.append((current_tick, track_idx, msg.note, 0, 'end'))
                    
                    elif msg.type == 'note_off':
                        # 音符结束
                        note_events.append((current_tick, track_idx, msg.note, 0, 'end'))
                    
                    elif msg.type == 'set_tempo':
                        # 记录速度变化 (微秒每拍)
                        track_info[track_idx]['tempo'] = msg.tempo
                    
                    elif msg.type == 'time_signature':
                        # 记录拍号变化
                        track_info[track_idx]['time_signature'] = (msg.numerator, msg.denominator)
                    
                    elif msg.type == 'key_signature':
                        # 记录调号变化
                        track_info[track_idx]['key_signature'] = msg.key
                    
                    elif msg.type == 'program_change':
                        # 记录乐器变化
                        track_info[track_idx]['program'] = msg.program
            
            # 按时间排序所有事件
            note_events.sort(key=lambda x: x[0])
            
            # 第二遍：匹配音符的开始和结束，计算时长
            active_notes_map = {}  # (track_idx, note_number) -> start_tick
            
            for event in note_events:
                abs_tick, track_idx, note_num, velocity, event_type = event
                key = (track_idx, note_num)
                
                if event_type == 'start':
                    # 记录音符开始
                    active_notes_map[key] = {
                        'start_tick': abs_tick,
                        'velocity': velocity,
                        'track_idx': track_idx
                    }
                elif event_type == 'end' and key in active_notes_map:
                    # 找到匹配的音符结束，计算时长
                    start_info = active_notes_map.pop(key)
                    duration_ticks = abs_tick - start_info['start_tick']
                    
                    if duration_ticks > 0:  # 过滤掉时长为0的音符
                        # 将tick转换为拍数 (beats)
                        duration_beats = duration_ticks / ticks_per_beat
                        start_beats = start_info['start_tick'] / ticks_per_beat
                        
                        # 获取当前音轨信息
                        track_data = track_info[track_idx]
                        
                        # 计算BPM
                        bpm = 60_000_000 / track_data['tempo']  # 微秒转BPM
                        
                        # 根据乐器program获取乐器名称
                        instrument_name = get_instrument_name(track_data['program'])
                        
                        notes.append({
                            'midi_pitch': note_num,
                            'note_name': midi_to_note(note_num),
                            'duration': float(duration_beats),
                            'start_time': float(start_beats),
                            'velocity': start_info['velocity'],
                            'matched': False,
                            'clip_id': None,
                            'pitch_shift': 0,
                            'track': track_idx,
                            'instrument': instrument_name,
                            'program': track_data['program'],
                            'tempo': bpm,
                            'time_signature': track_data['time_signature'],
                            'key_signature': track_data['key_signature'],
                            'source': 'midi'
                        })
            
            # 处理未结束的音符（如果MIDI文件没有相应的note_off）
            for key, start_info in active_notes_map.items():
                track_idx, note_num = key
                # 假设音符持续到文件末尾或给一个默认时长
                final_tick = max([event[0] for event in note_events]) if note_events else 0
                duration_ticks = final_tick - start_info['start_tick']
                
                if duration_ticks > 0:
                    duration_beats = duration_ticks / ticks_per_beat
                    start_beats = start_info['start_tick'] / ticks_per_beat
                    
                    track_data = track_info[track_idx]
                    instrument_name = get_instrument_name(track_data['program'])
                    
                    notes.append({
                        'midi_pitch': note_num,
                        'note_name': midi_to_note(note_num),
                        'duration': float(duration_beats),
                        'start_time': float(start_beats),
                        'velocity': start_info['velocity'],
                        'matched': False,
                        'clip_id': None,
                        'pitch_shift': 0,
                        'track': track_idx,
                        'instrument': instrument_name,
                        'program': track_data['program'],
                        'tempo': 60_000_000 / track_data['tempo'],
                        'time_signature': track_data['time_signature'],
                        'key_signature': track_data['key_signature'],
                        'source': 'midi'
                    })
            
            print(f"MIDI解析完成，共提取 {len(notes)} 个音符")
            
            # 检测并添加休止符
            notes = add_rests_to_midi(notes)
    
    except Exception as e:
        print(f"解析乐谱失败 {filepath}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 返回示例数据用于测试（仅当完全失败时）
        notes = [
            {'midi_pitch': 60, 'note_name': 'C4', 'duration': 1.0, 'start_time': 0.0, 
             'velocity': 64, 'matched': False, 'clip_id': None, 'pitch_shift': 0,
             'track': 0, 'instrument': 'piano', 'source': 'fallback'},
            {'midi_pitch': 62, 'note_name': 'D4', 'duration': 1.0, 'start_time': 1.0, 
             'velocity': 64, 'matched': False, 'clip_id': None, 'pitch_shift': 0,
             'track': 0, 'instrument': 'piano', 'source': 'fallback'},
            {'midi_pitch': 64, 'note_name': 'E4', 'duration': 2.0, 'start_time': 2.0, 
             'velocity': 64, 'matched': False, 'clip_id': None, 'pitch_shift': 0,
             'track': 0, 'instrument': 'piano', 'source': 'fallback'},
        ]
    
    # 按开始时间排序
    notes.sort(key=lambda x: x['start_time'])
    
    # 打印统计信息
    if notes:
        print(f"解析统计: 共 {len(notes)} 个音符")
        print(f"音高范围: {min(n['midi_pitch'] for n in notes if n['midi_pitch'] > 0)} 到 {max(n['midi_pitch'] for n in notes)}")
        print(f"时间范围: {notes[0]['start_time']:.2f} 到 {notes[-1]['start_time'] + notes[-1]['duration']:.2f} 拍")
        
        # 按音轨统计
        if any(n['track'] > 0 for n in notes):
            tracks = set(n['track'] for n in notes)
            print(f"音轨数: {len(tracks)}")
    
    return notes

def get_instrument_name(program: int) -> str:
    """根据MIDI程序号获取乐器名称"""
    # GM (General MIDI) 乐器列表 (0-127)
    gm_instruments = [
        "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", 
        "Honky-tonk Piano", "Electric Piano 1", "Electric Piano 2", "Harpsichord", 
        "Clavinet", "Celesta", "Glockenspiel", "Music Box", "Vibraphone", 
        "Marimba", "Xylophone", "Tubular Bells", "Dulcimer", "Drawbar Organ", 
        "Percussive Organ", "Rock Organ", "Church Organ", "Reed Organ", 
        "Accordion", "Harmonica", "Tango Accordion", "Acoustic Guitar (nylon)", 
        "Acoustic Guitar (steel)", "Electric Guitar (jazz)", "Electric Guitar (clean)", 
        "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar", 
        "Guitar harmonics", "Acoustic Bass", "Electric Bass (finger)", 
        "Electric Bass (pick)", "Fretless Bass", "Slap Bass 1", "Slap Bass 2", 
        "Synth Bass 1", "Synth Bass 2", "Violin", "Viola", "Cello", "Contrabass", 
        "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani", 
        "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2", 
        "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit", "Trumpet", 
        "Trombone", "Tuba", "Muted Trumpet", "French Horn", "Brass Section", 
        "Synth Brass 1", "Synth Brass 2", "Soprano Sax", "Alto Sax", "Tenor Sax", 
        "Baritone Sax", "Oboe", "English Horn", "Bassoon", "Clarinet", "Piccolo", 
        "Flute", "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi", "Whistle", 
        "Ocarina", "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", 
        "Lead 4 (chiff)", "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", 
        "Lead 8 (bass + lead)", "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", 
        "Pad 4 (choir)", "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", 
        "Pad 8 (sweep)", "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", 
        "FX 4 (atmosphere)", "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", 
        "FX 8 (sci-fi)", "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", 
        "Bag pipe", "Fiddle", "Shanai", "Tinkle Bell", "Agogo", "Steel Drums", 
        "Woodblock", "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal", 
        "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet", 
        "Telephone Ring", "Helicopter", "Applause", "Gunshot"
    ]
    
    if 0 <= program < 128:
        return gm_instruments[program]
    return f"Unknown ({program})"

def add_rests_to_midi(notes: List[Dict]) -> List[Dict]:
    """在MIDI音符之间检测并添加休止符"""
    if not notes:
        return notes
    
    notes_with_rests = []
    notes.sort(key=lambda x: (x['track'], x['start_time']))
    
    # 按音轨分组处理
    tracks = {}
    for note in notes:
        track_num = note['track']
        if track_num not in tracks:
            tracks[track_num] = []
        tracks[track_num].append(note)
    
    # 为每个音轨添加休止符
    for track_num, track_notes in tracks.items():
        track_notes.sort(key=lambda x: x['start_time'])
        
        current_time = 0.0
        
        for note in track_notes:
            # 如果当前时间和音符开始时间有间隔，添加休止符
            if note['start_time'] > current_time:
                rest_duration = note['start_time'] - current_time
                
                notes_with_rests.append({
                    'midi_pitch': -1,  # 休止符标识
                    'note_name': 'REST',
                    'duration': float(rest_duration),
                    'start_time': float(current_time),
                    'velocity': 0,
                    'matched': False,
                    'clip_id': None,
                    'pitch_shift': 0,
                    'track': track_num,
                    'instrument': 'rest',
                    'program': -1,
                    'tempo': note.get('tempo', 120),
                    'time_signature': note.get('time_signature', (4, 4)),
                    'key_signature': note.get('key_signature', 'C'),
                    'source': 'midi_rest'
                })
            
            notes_with_rests.append(note)
            current_time = note['start_time'] + note['duration']
    
    # 重新按时间排序
    notes_with_rests.sort(key=lambda x: x['start_time'])
    return notes_with_rests


def find_best_match_for_note(target_midi: int, tolerance_cents: float = 50.0, 
                           use_confidence_weight: bool = True) -> Tuple[Optional[Dict], float]:
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
    best_score = -float('inf')  # 使用评分系统，分数越高越好
    
    # 2. 确定搜索范围：目标音高附近 ± (容差/100 + 1) 个半音
    search_semitones = int(tolerance_cents / 100) + 2
    lower_bound = target_midi - search_semitones
    upper_bound = target_midi + search_semitones
    
    # 3. 在索引的邻近键中搜索
    for search_midi in range(lower_bound, upper_bound + 1):
        if search_midi not in index:
            continue
        
        for clip_data in index[search_midi]:
            clip = clip_data['clip']
            clip_exact_midi = clip_data['exact_midi']
            confidence = clip_data['confidence']
            
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
        # 这里可以保留你原有的线性搜索逻辑作为fallback，但使用索引通常能找到
        print(f"[匹配警告] 未在容差 {tolerance_cents} 音分内找到 MIDI {target_midi} 的匹配，返回最接近的。")
        # 简单实现：遍历所有片段找最接近的
        available_clips = clip_manager.get_all_clips()
        best_distance = float('inf')
        for clip in available_clips:
            note_info = clip.get('note_info', {})
            if note_info and note_info.get('frequency'):
                clip_freq = note_info['frequency']
                clip_midi = freq_to_midi(clip_freq)
                semitones_diff = target_midi - clip_midi
                cents_diff = abs(semitones_diff * 100)
                if cents_diff < best_distance:
                    best_distance = cents_diff
                    best_clip = clip
                    best_semitones = semitones_diff
    
    return best_clip, best_semitones

# 可选：当clip_manager的片段列表更新时，清除缓存以重建索引
def clear_clip_index_cache():
    """当添加或删除音频片段后，调用此函数清除索引缓存"""
    global _clip_index_cache
    _clip_index_cache = None
    print("音频片段索引缓存已清除，将在下次匹配时重建。")

def auto_generate_music_from_score(score_file, tempo=120, tolerance_cents=20.0, use_pitch_shift=True):
    """
    自动从乐谱生成音乐的主函数
    """
    if not score_file:
        return None, "请先上传乐谱文件", [], "❌ 未上传乐谱"
    
    try:
        generation_status = "🔄 开始解析乐谱..."
        yield None, generation_status, [], "解析中..."
        
        # 1. 解析乐谱
        notes = parse_score_notes(score_file)
        if not notes:
            return None, "❌ 未能从乐谱中解析出音符", [], "解析失败"
        
        generation_status = f"✅ 解析完成，共 {len(notes)} 个音符\n🔄 开始匹配音频片段..."
        yield None, generation_status, [], "匹配中..."
        
        # 2. 匹配音频片段
        sr = config.sample_rate
        beat_duration = 60.0 / tempo
        match_details = []
        
        # 为每个音符匹配片段
        for i, note in enumerate(notes):
            target_midi = note['midi_pitch']
            
            # >>> 修改点1：优先处理休止符 <<<
            if target_midi == -1:
                note['matched'] = True
                note['is_rest'] = True
                match_details.append([
                    f"音符{i+1}",
                    note['note_name'],
                    f"休止符 ({note['duration']:.2f}拍)",
                    "N/A",
                    "⏸️ 休止",
                    note.get('track', 0),  # 展示音轨信息
                    note.get('instrument', 'rest')
                ])
                continue  # 跳过后续匹配逻辑
            
            # 寻找最佳匹配（仅针对普通音符）
            best_clip, semitones_diff = find_best_match_for_note(target_midi, tolerance_cents)
            
            if best_clip:
                note['matched'] = True
                note['clip_id'] = best_clip['id']
                note['pitch_shift'] = semitones_diff if use_pitch_shift else 0
                
                match_status = "✅ 完全匹配" if abs(semitones_diff) < 0.1 else f"🔄 需变调 {semitones_diff:+.1f} 半音"
                
                match_details.append([
                    f"音符{i+1}",
                    note['note_name'],
                    f"片段{best_clip['id']} ({best_clip.get('note_info', {}).get('note', '未知')})",
                    f"{semitones_diff:+.1f}" if use_pitch_shift else "0",
                    match_status,
                    note.get('track', 0),  # 新增：展示音轨信息
                    note.get('instrument', 'unknown')  # 新增：展示乐器信息
                ])
            else:
                note['matched'] = False
                match_details.append([
                    f"音符{i+1}",
                    note['note_name'],
                    "无可用片段",
                    "N/A",
                    "❌ 未匹配",
                    note.get('track', 0),
                    note.get('instrument', 'unknown')
                ])
        
        # 统计匹配结果（仅统计普通音符，排除休止符）
        valid_notes = [n for n in notes if n.get('midi_pitch', 0) != -1]
        matched_count = sum(1 for n in valid_notes if n['matched'])
        total_valid_notes = len(valid_notes)
        match_rate = matched_count / total_valid_notes * 100 if total_valid_notes > 0 else 0
        
        generation_status = f"✅ 匹配完成: {matched_count}/{total_valid_notes} 个可匹配音符 ({match_rate:.1f}%)\n🔄 开始处理音频..."
        yield None, generation_status, match_details, "处理中..."
        
        # 3. 处理音频片段
        processed_clips = {}
        audio_segments = []
        
        for i, note in enumerate(notes):
            # >>> 修改点2：优先处理休止符 <<<
            if note.get('is_rest') or note['midi_pitch'] == -1:
                # 生成静音片段
                silence_duration = note['duration'] * beat_duration
                silence_samples = int(silence_duration * sr)
                audio_segments.append((note['start_time'], np.zeros(silence_samples, dtype=np.float32)))
                continue
            
            # 处理未匹配的普通音符（生成静音）
            if not note['matched']:
                silence_duration = note['duration'] * beat_duration
                silence_samples = int(silence_duration * sr)
                audio_segments.append((note['start_time'], np.zeros(silence_samples, dtype=np.float32)))
                continue
            
            # 处理已匹配的普通音符
            clip_id = note['clip_id']
            semitones = note['pitch_shift']
            
            # 如果已处理过相同变调的片段，直接重用
            cache_key = f"{clip_id}_{semitones}"
            if cache_key not in processed_clips:
                # 加载原始音频
                clip = clip_manager.clips[clip_id]
                y, clip_sr = sf.read(clip['filepath'])
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                
                # 重采样到目标采样率
                if clip_sr != sr:
                    y = librosa.resample(y, orig_sr=clip_sr, target_sr=sr)
                
                # 变调处理
                if use_pitch_shift and abs(semitones) > 0.1:
                    y = pitch_shift(y, sr, semitones)
                
                processed_clips[cache_key] = y
            
            # 获取处理后的音频
            y_processed = processed_clips[cache_key].copy()
            
            # 时间拉伸以匹配音符时长
            target_duration = note['duration'] * beat_duration
            current_duration = len(y_processed) / sr
            
            if abs(current_duration - target_duration) > 0.01:  # 10ms容差
                rate = current_duration / target_duration
                rate = np.clip(rate, 0.5, 2.0)  # 限制拉伸范围
                y_processed = librosa.effects.time_stretch(y_processed, rate=rate)
            
            # 调整到精确长度
            target_samples = int(target_duration * sr)
            if len(y_processed) > target_samples:
                y_processed = y_processed[:target_samples]
            else:
                y_processed = np.pad(y_processed, (0, target_samples - len(y_processed)), mode='constant')
            
            # 应用音量调整（基于velocity）
            velocity_factor = note['velocity'] / 127.0
            y_processed *= velocity_factor * 0.7  # 避免过载
            
            # 添加淡入淡出
            y_processed = apply_fade(y_processed, sr, fade_in=0.02, fade_out=0.05)
            
            audio_segments.append((note['start_time'], y_processed))
            
            # 每处理10个片段更新一次状态
            if i % 10 == 0 and i > 0:
                processed_count = len([n for n in notes[:i+1] if not n.get('is_rest') and n['midi_pitch'] != -1])
                generation_status = f"✅ 已处理 {processed_count}/{total_valid_notes} 个音符\n🔄 继续处理音频..."
                yield None, generation_status, match_details, "处理中..."
        
        generation_status = f"✅ 音频处理完成，共 {len(audio_segments)} 个音频片段\n🔄 开始拼接音乐..."
        yield None, generation_status, match_details, "拼接中..."
        
        # 4. 拼接所有音频片段 - 关键修复部分
        # 计算总时长（以秒为单位）
        max_end_time_seconds = 0
        generation_status = f"🔄 正在计算总时长..."
        yield None, generation_status, match_details, "计算时长中..."
        
        for start_time, segment in audio_segments:
            segment_duration = len(segment) / sr
            end_time_seconds = start_time * beat_duration + segment_duration
            if end_time_seconds > max_end_time_seconds:
                max_end_time_seconds = end_time_seconds
        
        generation_status = f"✅ 总时长计算完成: {max_end_time_seconds:.2f}秒\n🔄 正在分配内存..."
        yield None, generation_status, match_details, "分配内存中..."
        
        # 确保有足够的空间，加上0.5秒的余量
        total_samples = int(max_end_time_seconds * sr) + int(0.5 * sr)
        final_audio = np.zeros(total_samples, dtype=np.float32)
        
        generation_status = f"✅ 内存分配完成: {total_samples}个样本\n🔄 开始放置音频片段..."
        yield None, generation_status, match_details, "放置片段中..."
        
        # 按时间线放置音频片段
        placed_count = 0
        for i, (start_time, segment) in enumerate(audio_segments):
            start_sample = int(start_time * beat_duration * sr)
            end_sample = start_sample + len(segment)
            
            # 确保片段在范围内
            if start_sample < len(final_audio):
                # 计算实际结束位置
                end_actual = min(end_sample, len(final_audio))
                # 确保段长度正确
                segment_len = end_actual - start_sample
                if segment_len > 0:
                    # 使用叠加而不是覆盖
                    final_audio[start_sample:end_actual] += segment[:segment_len]
                    placed_count += 1
            
            # 每放置10个片段更新一次状态
            if i % 10 == 0 and i > 0:
                generation_status = f"🔄 已放置 {i+1}/{len(audio_segments)} 个片段..."
                yield None, generation_status, match_details, "放置片段中..."
        
        generation_status = f"✅ 片段放置完成: {placed_count}/{len(audio_segments)} 个片段\n🔄 正在归一化..."
        yield None, generation_status, match_details, "归一化中..."
        
        # 归一化
        final_audio = normalize_audio(final_audio)
        
        # 添加淡出效果，避免突然结束
        fade_out_samples = int(0.05 * sr)
        if fade_out_samples > 0 and fade_out_samples <= len(final_audio):
            fade_out_window = np.linspace(1, 0, fade_out_samples)
            final_audio[-fade_out_samples:] *= fade_out_window
        
        generation_status = f"✅ 音频处理完成\n🔄 正在生成报告..."
        yield None, generation_status, match_details, "生成报告中..."
        
        # 5. 生成报告
        report = f"""
        ## 🎵 音乐生成报告
        
        ### 基本信息
        - **乐谱文件**: {os.path.basename(score_file)}
        - **音符总数**: {len(notes)} (含休止符)
        - **可匹配音符**: {total_valid_notes} (不含休止符)
        - **演奏速度**: {tempo} BPM
        - **总时长**: {total_samples/sr:.2f} 秒
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
            if note['matched']:
                clip_id = note['clip_id']
                used_clips[clip_id] = used_clips.get(clip_id, 0) + 1
        
        for clip_id, count in used_clips.items():
            clip = clip_manager.clips[clip_id]
            note_name = clip.get('note_info', {}).get('note', '未知')
            report += f"- **片段{clip_id}** ({note_name}): 使用 {count} 次\n"
        
        # >>> 修改点3：添加音轨与乐器统计 <<<
        report += f"\n### 音轨与乐器信息\n"
        # 统计音轨
        tracks_used = set(n.get('track', 0) for n in notes if n.get('track') is not None)
        report += f"- **使用音轨数**: {len(tracks_used)} 个\n"
        
        # 按音轨统计音符
        if len(tracks_used) > 1:
            report += f"- **各音轨音符分布**:\n"
            for track_num in sorted(tracks_used):
                track_notes = [n for n in notes if n.get('track', 0) == track_num and n['midi_pitch'] != -1]
                if track_notes:
                    instr = track_notes[0].get('instrument', 'unknown')
                    report += f"  - 音轨{track_num} ({instr}): {len(track_notes)} 个音符\n"
        
        # 统计乐器（仅统计非休止符）
        instruments_used = {}
        for note in valid_notes:
            instr = note.get('instrument', 'unknown')
            instruments_used[instr] = instruments_used.get(instr, 0) + 1
        
        if instruments_used:
            report += f"- **乐器分布**:\n"
            for instr, count in sorted(instruments_used.items(), key=lambda x: x[1], reverse=True):
                report += f"  - {instr}: {count} 个音符\n"
        
        report += f"\n### 调试信息\n"
        report += f"- **最大结束时间**: {max_end_time_seconds:.2f} 秒\n"
        report += f"- **总样本数**: {total_samples} 个\n"
        report += f"- **实际时长**: {len(final_audio)/sr:.2f} 秒\n"
        
        # 如果有原始MIDI速度信息，显示对比
        tempos = set(n.get('tempo') for n in notes if n.get('tempo'))
        if len(tempos) == 1:
            original_tempo = list(tempos)[0]
            report += f"- **原始乐谱速度**: {original_tempo:.0f} BPM\n"
            report += f"- **实际使用速度**: {tempo} BPM\n"
        
        report += f"\n⏱️ **生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # 保存生成的音乐
        output_filename = f"auto_composition_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        output_path = os.path.join(config.output_dir, output_filename)
        sf.write(output_path, final_audio, sr)
        
        generation_status = f"✅ 音乐生成完成！\n📁 已保存至: {output_filename}"
        
        yield (sr, final_audio), report, match_details, generation_status
        
    except Exception as e:
        error_msg = f"❌ 生成过程中出错: {str(e)}"
        print(f"生成音乐失败: {e}")
        import traceback
        traceback.print_exc()
        yield None, error_msg, [], "生成失败"

# ========= 创建Gradio界面 =========

def build_advanced_ui():
    with gr.Blocks(title="高级音频处理与音乐制作系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎵 高级音频处理与音乐制作系统
        
        ## 功能介绍
        1. **音频识别**：自动检测音频频率并转换为音名
        2. **音频处理**：支持变速、变调、淡入淡出
        3. **音乐制作**：根据乐谱或手动编排制作音乐
        4. **频谱分析**：可视化音频特征
        """)
        
        with gr.Tabs():
            with gr.TabItem("🎤 音频上传与识别"):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            label="上传音频文件",
                            type="filepath"
                        )
                        target_note = gr.Textbox(
                            label="目标音高（可选）",
                            placeholder="例如：C4, D#4, Gb5",
                            value=""
                        )
                        
                        with gr.Row():
                            auto_detect = gr.Checkbox(
                                label="自动检测音高",
                                value=True
                            )
                            analysis_mode = gr.Radio(
                                choices=["simple", "enhanced"],
                                label="分析模式",
                                value="simple",
                                info="简单模式：频谱图+波形图 | 增强模式：多种分析图表"
                            )
                        
                        btn_analyze = gr.Button("分析音频", variant="primary")
                        
                    with gr.Column(scale=2):
                        result_text = gr.Markdown(label="分析结果")
                        clip_id_output = gr.Number(
                            label="片段ID",
                            visible=False
                        )
                        spectrogram = gr.Plot(
                            label="频谱图分析"
                        )
                        enhanced_analysis = gr.Plot(
                            label="增强分析",
                            visible=False
                        )
                        
                        def toggle_analysis(analysis_mode):
                            if analysis_mode == "enhanced":
                                return gr.Plot(visible=True)
                            else:
                                return gr.Plot(visible=False)
                        
                        analysis_mode.change(
                            fn=toggle_analysis,
                            inputs=[analysis_mode],
                            outputs=[enhanced_analysis]
                        )
                
                btn_analyze.click(
                    fn=handle_audio_upload,
                    inputs=[audio_input, target_note, auto_detect, analysis_mode],
                    outputs=[result_text, clip_id_output, spectrogram, enhanced_analysis]
                )
                
                # 频谱图说明
                with gr.Accordion("📊 频谱图解读指南", open=False):
                    gr.Markdown("""
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
                    """)
            
            with gr.TabItem("🎛️ 音频处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        clip_id_input = gr.Number(
                            label="片段ID",
                            value=0,
                            precision=0
                        )
                        operation = gr.Radio(
                            choices=["time_stretch", "pitch_shift"],
                            label="处理类型",
                            value="time_stretch"
                        )
                        value_input = gr.Slider(
                            label="参数值",
                            minimum=0.1,
                            maximum=5.0,
                            value=1.0,
                            step=0.1,
                            visible=True
                        )
                        
                        def update_slider(operation):
                            if operation == "time_stretch":
                                return gr.Slider(
                                    minimum=0.1,
                                    maximum=5.0,
                                    value=1.0,
                                    step=0.1,
                                    label="目标时长（秒）"
                                )
                            else:
                                return gr.Slider(
                                    minimum=-12,
                                    maximum=12,
                                    value=0,
                                    step=0.5,
                                    label="半音移动"
                                )
                        
                        operation.change(
                            fn=update_slider,
                            inputs=[operation],
                            outputs=[value_input]
                        )
                        
                        btn_process = gr.Button("处理音频", variant="primary")
                        process_result = gr.Markdown(label="处理结果")
                    
                    with gr.Column(scale=2):
                        audio_preview = gr.Audio(
                            label="处理结果预览",
                            type="numpy"
                        )
                
                btn_process.click(
                    fn=process_audio_clip,
                    inputs=[clip_id_input, operation, value_input],
                    outputs=[process_result, audio_preview]
                )
            
            with gr.TabItem("🎹 音乐制作"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("""
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
                        """)
                        
                        clip_assignments = gr.Textbox(
                            label="片段分配",
                            placeholder="格式: 拍数:片段ID, 拍数:片段ID,...",
                            lines=3
                        )
                        tempo_input = gr.Slider(
                            label="速度 (BPM)",
                            minimum=40,
                            maximum=240,
                            value=120,
                            step=10
                        )
                        btn_compose = gr.Button("生成音乐", variant="primary")
                        compose_result = gr.Markdown(label="生成结果")
                    
                    with gr.Column(scale=2):
                        composition_audio = gr.Audio(
                            label="生成音乐",
                            type="numpy"
                        )
                
                btn_compose.click(
                    fn=generate_music_from_clips,
                    inputs=[clip_assignments, tempo_input],
                    outputs=[compose_result, composition_audio]
                )
            
            with gr.TabItem("📋 片段管理"):
                def update_clips_table():
                    clips = clip_manager.get_all_clips()
                    table_data = []
                    for clip in clips:
                        note_info = clip.get('note_info', {})
                        table_data.append([
                            clip['id'],
                            clip['filename'],
                            note_info.get('note', '未知'),
                            f"{note_info.get('frequency', 0):.1f}" if note_info.get('frequency') else '未知',
                            f"{note_info.get('cents', 0):+.1f}" if note_info.get('cents') is not None else '',
                            f"{clip['duration']:.2f}",
                            clip['created_at']
                        ])
                    return table_data
                
                with gr.Row():
                    clips_table = gr.Dataframe(
                        headers=["ID", "文件名", "音名", "频率", "偏差", "时长", "创建时间"],
                        label="所有音频片段",
                        datatype=["number", "str", "str", "str", "str", "str", "str"],
                        row_count=10,
                        col_count=7,
                        interactive=False
                    )
                
                with gr.Row():
                    btn_refresh = gr.Button("刷新列表")
                    delete_clip_id = gr.Number(
                        label="删除片段ID",
                        value=0,
                        precision=0
                    )
                    btn_delete = gr.Button("删除片段", variant="stop")
                
                with gr.Row():
                    btn_cleanup = gr.Button("清理孤立文件", variant="secondary")

                def cleanup_orphaned():
                    clip_manager.cleanup_orphaned_files()
                    return "已清理孤立文件", update_clips_table()

                btn_cleanup.click(
                    fn=cleanup_orphaned,
                    inputs=[],
                    outputs=[compose_result, clips_table]
                )
                
                def delete_selected_clip(clip_id):
                    success = clip_manager.delete_clip(int(clip_id))
                    if success:
                        return f"✅ 已删除片段 {clip_id}", update_clips_table()
                    else:
                        return f"❌ 删除失败，片段 {clip_id} 不存在", update_clips_table()
                
                btn_refresh.click(
                    fn=update_clips_table,
                    inputs=[],
                    outputs=[clips_table]
                )
                
                btn_delete.click(
                    fn=delete_selected_clip,
                    inputs=[delete_clip_id],
                    outputs=[compose_result, clips_table]
                )

            build_music_composition_tab()
        gr.Markdown("""
        ## 📚 使用说明
        
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
        """)
    
    return demo

# ========= 主程序 =========

if __name__ == "__main__":
    # 创建示例配置文件
    config_data = {
        "audio_settings": {
            "sample_rate": 22050,
            "min_freq": 32.70,
            "max_freq": 4186.01,
            "tempo": 120
        },
        "detection_settings": {
            "silence_threshold_db": 40,
            "confidence_threshold": 0.7,
            "min_clip_duration": 0.05
        },
        "processing_settings": {
            "time_stretch_range": [0.5, 2.0],
            "pitch_shift_range": [-12, 12],
            "fade_in": 0.01,
            "fade_out": 0.01
        }
    }
    
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    
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
        show_error=True
    )