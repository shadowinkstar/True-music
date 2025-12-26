import os

import matplotlib


def setup_chinese_fonts() -> None:
    """尝试设置中文字体，避免中文显示为方块。"""
    try:
        # 尝试设置中文字体
        font_list = []

        # Windows字体路径
        if os.name == "nt":
            font_dirs = [
                "C:/Windows/Fonts",  # Windows系统字体
                os.path.expanduser("~/.fonts"),  # 用户字体
            ]
            # 常见中文字体名称
            chinese_fonts = [
                "msyh.ttc",  # 微软雅黑
                "simhei.ttf",  # 黑体
                "simsun.ttc",  # 宋体
                "simkai.ttf",  # 楷体
                "STHeiti Light.ttc",  # 华文黑体（Mac）
                "PingFang.ttc",  # 苹方（Mac）
            ]
        else:
            # Linux/Mac字体路径
            font_dirs = [
                "/usr/share/fonts",
                "/usr/local/share/fonts",
                os.path.expanduser("~/.fonts"),
                os.path.expanduser("~/Library/Fonts"),  # Mac
            ]
            chinese_fonts = [
                "wqy-microhei.ttc",  # 文泉驿微米黑
                "NotoSansCJK-Regular.ttc",  # Noto字体
                "SourceHanSansSC-Regular.otf",  # 思源黑体
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
            font_name = matplotlib.font_manager.FontProperties(
                fname=available_fonts[0]
            ).get_name()
            matplotlib.rcParams["font.sans-serif"] = [font_name]
            matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
            print(f"已设置中文字体: {font_name}")
        else:
            print("未找到中文字体，将使用默认字体")

    except Exception as exc:
        print(f"设置字体时出错: {exc}")
        # 使用默认字体，但尝试设置支持中文的字体
        matplotlib.rcParams["font.sans-serif"] = [
            "DejaVu Sans",
            "Microsoft YaHei",
            "SimHei",
            "Arial Unicode MS",
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False
