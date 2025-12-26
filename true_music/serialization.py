import numpy as np


def convert_to_serializable(obj):
    """将numpy类型转换为Python原生类型以便JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    return obj
