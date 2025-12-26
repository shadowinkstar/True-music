import librosa
import matplotlib.pyplot as plt
import numpy as np


def create_spectrogram(
    y: np.ndarray, sr: int, detected_freq: float = None
) -> plt.Figure:
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
    img = librosa.display.specshow(
        S_dB, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel", cmap="viridis"
    )

    plt.colorbar(img, format="%+2.0f dB", ax=ax1)
    plt.title("音频频谱图 (Mel Spectrogram)", fontsize=14, fontweight="bold")
    plt.xlabel("时间 (秒)")
    plt.ylabel("频率 (Hz) - Mel刻度")

    # 标记检测到的频率
    if detected_freq:
        # 将频率转换为mel刻度
        mel_freq = librosa.hz_to_mel(detected_freq)
        plt.axhline(
            y=mel_freq,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"检测基频: {detected_freq:.1f} Hz",
        )

        # 在右侧显示频率值
        plt.text(
            plt.xlim()[1] * 1.02,
            mel_freq,
            f"{detected_freq:.0f} Hz",
            color="red",
            va="center",
            fontsize=10,
        )

        plt.legend(loc="upper right")

    # 添加网格，提高可读性
    plt.grid(True, alpha=0.3, linestyle="--")

    # 在下方显示波形图
    ax2 = plt.subplot(2, 1, 2)
    time = np.linspace(0, len(y) / sr, len(y))
    plt.plot(time, y, color="blue", alpha=0.7, linewidth=0.5)
    plt.fill_between(time, y, 0, alpha=0.3, color="blue")

    plt.title("音频波形", fontsize=14, fontweight="bold")
    plt.xlabel("时间 (秒)")
    plt.ylabel("振幅")
    plt.grid(True, alpha=0.3, linestyle="--")

    # 设置x轴范围一致
    ax1.set_xlim([0, len(y) / sr])
    ax2.set_xlim([0, len(y) / sr])

    plt.tight_layout()

    return plt.gcf()


def create_enhanced_analysis(y: np.ndarray, sr: int, detected_info: dict) -> plt.Figure:
    """
    创建增强分析图，包含多种可视化
    """
    fig = plt.figure(figsize=(15, 10))

    # 1. 频谱图
    ax1 = plt.subplot(3, 2, 1)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title("频谱图")

    # 2. 波形图
    ax2 = plt.subplot(3, 2, 2)
    time = np.linspace(0, len(y) / sr, len(y))
    plt.plot(time, y, color="blue", alpha=0.7, linewidth=0.5)
    plt.fill_between(time, y, 0, alpha=0.3, color="blue")
    plt.title("波形图")
    plt.xlabel("时间 (秒)")
    plt.ylabel("振幅")
    plt.grid(True, alpha=0.3)

    # 3. 频谱图（线性频率）
    ax3 = plt.subplot(3, 2, 3)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("频谱图（对数频率）")

    # 4. 基频轨迹（如果有）
    if detected_info.get("frequency"):
        ax4 = plt.subplot(3, 2, 4)

        # 计算基频轨迹
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )

        times = librosa.times_like(f0, sr=sr)

        plt.plot(times, f0, label="基频轨迹", color="red", linewidth=2)
        plt.axhline(
            y=detected_info["frequency"],
            color="green",
            linestyle="--",
            label=f"检测频率: {detected_info['frequency']:.1f} Hz",
        )
        plt.title("基频轨迹")
        plt.xlabel("时间 (秒)")
        plt.ylabel("频率 (Hz)")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 5. 频谱质心
    ax5 = plt.subplot(3, 2, 5)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    times = librosa.times_like(spectral_centroids, sr=sr)
    plt.plot(times, spectral_centroids, color="purple")
    plt.title("频谱质心")
    plt.xlabel("时间 (秒)")
    plt.ylabel("频率 (Hz)")
    plt.grid(True, alpha=0.3)

    # 6. 过零率
    ax6 = plt.subplot(3, 2, 6)
    zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
    times = librosa.times_like(zero_crossings, sr=sr)
    plt.plot(times, zero_crossings, color="orange")
    plt.title("过零率")
    plt.xlabel("时间 (秒)")
    plt.ylabel("过零率")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
