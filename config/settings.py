# 配置文件：所有固定路径/变量名集中管理
import os

# 项目根路径（自动适配，避免硬编码）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模型路径
#MODEL_PATH = os.path.join(PROJECT_ROOT, "Prediction_model", "ACd_model.pkl")

# 配置项统一管理
CONFIG = {
    "MODEL_PATH": os.path.join(PROJECT_ROOT, "Prediction_model", "ACd_model.pkl"),
#r"D:\硕士科研\Programming\github\batch_ACd_analysis\final\Prediction_model\ACd_model.pkl",
    "FEATURE_COLS": ["TCd", "pH", "SM", "PSS", "CEC", "SOM"],
    "TARGET_COL": "ACd",
    "LOG_ADD_CONSTANT": 0.00000001,
    "STATS_DECIMALS": 3,
    "STANDARD_DERIVE_CONFIG": {
        "var_stats": {
            "SOM": {"n_points": 15},
            "pH": {"n_points": 15},
            "PSS": {"n_points": 15},
            "CEC": {"default_median": 25},
            "TCd": {"fixed_original_value": 0.3},
            "SM": {"default_median": 30}
        },
        "plot": {
            "figsize": (3, 2),
            "dpi": 600,
            "cmap": "viridis",
            "alpha": 0.7,
            "scatter_size": 10,
            "edgecolors": "black",
            "linewidth": 0.2,
            "colorbar_pad": 0.00000000001,
            "colorbar_shrink": 0.6,
            "colorbar_aspect": 20,
            "view_elev": 25,
            "view_azim": 80
        },
        "decimal": 3
    }
}