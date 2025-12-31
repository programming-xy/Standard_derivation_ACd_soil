import numpy as np
import pandas as pd
import os
import pickle
from config.settings import CONFIG

def log_transform(data, add_constant=CONFIG["LOG_ADD_CONSTANT"]):
    """对数变换（适配DataFrame/Series/数组，避免log(0)）"""
    data_np = np.array(data)
    if np.any(data_np + add_constant <= 0):
        raise ValueError(f"对数化失败：存在数据使得 data + {add_constant} <= 0！")
    log_data = np.log(data_np + add_constant)
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(log_data, columns=data.columns, index=data.index)
    elif isinstance(data, pd.Series):
        return pd.Series(log_data, index=data.index, name=data.name)
    else:
        return log_data

def exp_transform(log_data, add_constant=CONFIG["LOG_ADD_CONSTANT"]):
    """指数还原（对数化逆操作）"""
    log_data_np = np.array(log_data)
    exp_data = np.exp(log_data_np) - add_constant
    if isinstance(log_data, pd.DataFrame):
        return pd.DataFrame(exp_data, columns=log_data.columns, index=log_data.index)
    elif isinstance(log_data, pd.Series):
        return pd.Series(exp_data, index=log_data.index, name=log_data.name)
    else:
        return exp_data

def load_model(model_path=CONFIG["MODEL_PATH"]):
    """加载训练好的XGB模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在：{model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def read_data(file):
    """读取上传文件（兼容CSV/XLSX，校验必需列）"""
    try:
        if file.name.endswith('.csv'):
            encodings = ['utf-8', 'gbk', 'gb2312', 'ansi', 'utf-8-sig']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise ValueError("无法识别CSV文件编码")
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            raise ValueError("仅支持CSV/XLSX文件")
        
        required_cols = CONFIG["FEATURE_COLS"] + [CONFIG["TARGET_COL"]]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"文件缺少必需列：{missing_cols}")
        if df[required_cols].isnull().any().any():
            raise ValueError(f"必需列包含空值，请清洗数据！")
        
        return df.reset_index(drop=True)
    except Exception as e:
        raise RuntimeError(f"读取文件失败：{str(e)}")