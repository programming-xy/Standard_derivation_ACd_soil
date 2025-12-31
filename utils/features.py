import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as mcolors
from sklearn.metrics import r2_score
from config.settings import CONFIG
from utils.utils import log_transform, exp_transform, read_data

def batch_analyze_files(uploaded_files, model):
    """批量处理上传文件，返回：汇总结果、数据统计、对数R²"""
    all_results = []
    data_stats = None
    r2_log_scale = None
    
    for file in uploaded_files:
        try:
            df = read_data(file)
            st.subheader(f"📄 The processing file：{file.name}")
            st.success(f"✅ Data reading successful | Total record count: {len(df):,}")
            
            # 描述性统计
            stats_df = df[CONFIG["FEATURE_COLS"] + [CONFIG["TARGET_COL"]]].describe()
            stats_df = round(stats_df, CONFIG["STATS_DECIMALS"])
            with st.expander(f"📊 {file.name} statistics", expanded=False):
                st.dataframe(stats_df, use_container_width=True)
            
            # 提取变量统计
            data_stats = {
                "pH": {"min": stats_df.loc['min', 'pH'], "max": stats_df.loc['max', 'pH']},
                "PSS": {"min": stats_df.loc['min', 'PSS'], "max": stats_df.loc['max', 'PSS']},
                "SOM": {"min": stats_df.loc['min', 'SOM'], "max": stats_df.loc['max', 'SOM']},
                "CEC": {"median": stats_df.loc['50%', 'CEC'] if 'CEC' in stats_df else CONFIG["STANDARD_DERIVE_CONFIG"]["var_stats"]["CEC"]["default_median"]},
                "SM": {"median": stats_df.loc['50%', 'SM'] if 'SM' in stats_df else CONFIG["STANDARD_DERIVE_CONFIG"]["var_stats"]["SM"]["default_median"]},
                "TCd": {"fixed_original": CONFIG["STANDARD_DERIVE_CONFIG"]["var_stats"]["TCd"]["fixed_original_value"]}
            }
            
            # 模型预测
            X = df[CONFIG["FEATURE_COLS"]].copy()
            X_log = log_transform(X)
            y_true = df[CONFIG["TARGET_COL"]].copy()
            y_true_log = log_transform(y_true)
            
            y_pred_log = model.predict(X_log)
            y_pred_ori = exp_transform(y_pred_log)
            
            # 计算指标
            batch_pred_median = round(np.median(y_pred_ori), CONFIG["STATS_DECIMALS"])
	     r2_log_scale = round(r2_score(y_true_log, y_pred_log), CONFIG["STATS_DECIMALS"])
	     r2_ori_scale = round(r2_score(y_true, y_pred_ori), CONFIG["STATS_DECIMALS"])
            
            # 展示结果
            col1, col2 = st.columns(2)
            with col1:
                st.metric("The predictive median of ACd", f"{batch_pred_median:.4f}")
            with col2:
                st.metric("R²", r2_log_scale)
            #with col3:
                #st.metric("原始尺度R²", r2_ori_scale)
            
            # 下载结果
            pred_df = df.copy()
            pred_df['ACd_predictive values (logarithmic scale)'] = round(y_pred_log, CONFIG["STATS_DECIMALS"])
            pred_df['ACd_predictive values (original scale)'] = round(y_pred_ori, CONFIG["STATS_DECIMALS"])
            csv_data = pred_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label=f"💾 Downloading {file.name} prediction results",
                data=csv_data,
                file_name=f"{os.path.splitext(file.name)[0]}_prediction results.csv",
                mime="text/csv",
                key=f"download_{file.name}"
            )
            
            all_results.append({
                "File_name": file.name,
                "Records": len(df),
                "Predictive median of ACd": batch_pred_median,
                "R²": r2_log_scale,
                #"原始尺度R²": r2_ori_scale
            })
        except Exception as e:
            st.error(f"❌ {file.name} failed to process：{str(e)}")
            continue
    
    if all_results:
        st.subheader("📈 The summary of batch processing")
        summary_df = pd.DataFrame(all_results)
        st.dataframe(summary_df, use_container_width=True)
        summary_csv = summary_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="💾 Downloading the batch summary results",
            data=summary_csv,
            file_name="The summary of batch analysis for ACd.csv",
            mime="text/csv"
        )
    
    return data_stats, r2_log_scale

def generate_3d_grid(data_stats):
    """生成3D网格数据"""
    pH_vals = np.linspace(data_stats["pH"]["min"], data_stats["pH"]["max"], CONFIG["STANDARD_DERIVE_CONFIG"]["var_stats"]["pH"]["n_points"])
    PSS_vals = np.linspace(data_stats["PSS"]["min"], data_stats["PSS"]["max"], CONFIG["STANDARD_DERIVE_CONFIG"]["var_stats"]["PSS"]["n_points"])
    SOM_vals = np.linspace(data_stats["SOM"]["min"], data_stats["SOM"]["max"], CONFIG["STANDARD_DERIVE_CONFIG"]["var_stats"]["SOM"]["n_points"])
    
    PSS_grid, pH_grid, SOM_grid = np.meshgrid(PSS_vals, pH_vals, SOM_vals, indexing='ij')
    
    TCd_fixed_ori = data_stats["TCd"]["fixed_original"]
    TCd_fixed_log = log_transform(TCd_fixed_ori)
    
    CEC_median_ori = data_stats["CEC"]["median"]
    CEC_median_log = log_transform(CEC_median_ori)
    
    SM_median_ori = data_stats["SM"]["median"]
    SM_median_log = log_transform(SM_median_ori)
    
    pH_flat = pH_grid.flatten()
    PSS_flat = PSS_grid.flatten()
    SOM_flat = SOM_grid.flatten()
    
    grid_data_log = pd.DataFrame({
        "TCd": np.full_like(pH_flat, TCd_fixed_log),
        "pH": log_transform(pH_flat),
        "SM": np.full_like(pH_flat, SM_median_log),
        "PSS": log_transform(PSS_flat),
        "CEC": np.full_like(pH_flat, CEC_median_log),
        "SOM": log_transform(SOM_flat)
    })[CONFIG["FEATURE_COLS"]]
    
    return {
        "grids": (pH_grid, PSS_grid, SOM_grid),
        "grid_data_log": grid_data_log,
        "var_ranges": {
            "pH": (data_stats["pH"]["min"], data_stats["pH"]["max"]),
            "PSS": (data_stats["PSS"]["min"], data_stats["PSS"]["max"]),
            "SOM": (data_stats["SOM"]["min"], data_stats["SOM"]["max"]),
            "TCd": (TCd_fixed_ori, TCd_fixed_log),
            "CEC": (CEC_median_ori, CEC_median_log),
            "SM": (SM_median_ori, SM_median_log)
        }
    }

def derive_acd_standard(model, data_stats, r2_log_scale):
    """推导ACd保护标准"""
    try:
        grid_result = generate_3d_grid(data_stats)
        grid_data_log = grid_result["grid_data_log"]
        pH_grid, PSS_grid, SOM_grid = grid_result["grids"]
        
        ACd_pred_log = model.predict(grid_data_log)
        ACd_pred_ori = exp_transform(ACd_pred_log)
        
        ACd_ori_flat = ACd_pred_ori.flatten()
        raw_median = round(np.median(ACd_ori_flat), CONFIG["STANDARD_DERIVE_CONFIG"]["decimal"])
        
        if not (0 < r2_log_scale <= 1):
            st.warning(f"⚠️ R²={r2_log_scale} is abnormal")
            corrected_median = raw_median
        else:
            corrected_median = round(raw_median / r2_log_scale, CONFIG["STANDARD_DERIVE_CONFIG"]["decimal"])
        
        # 绘图配置
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["axes.unicode_minus"] = False
        
        fig = plt.figure(figsize=CONFIG["STANDARD_DERIVE_CONFIG"]["plot"]["figsize"], 
                         dpi=CONFIG["STANDARD_DERIVE_CONFIG"]["plot"]["dpi"])
        ax = fig.add_subplot(111, projection='3d')
        
        norm = mcolors.Normalize(vmin=np.min(ACd_ori_flat), vmax=np.max(ACd_ori_flat))
        scatter = ax.scatter(
            pH_grid, PSS_grid, SOM_grid,
            c=ACd_ori_flat, cmap=CONFIG["STANDARD_DERIVE_CONFIG"]["plot"]["cmap"],
            norm=norm, alpha=CONFIG["STANDARD_DERIVE_CONFIG"]["plot"]["alpha"],
            s=CONFIG["STANDARD_DERIVE_CONFIG"]["plot"]["scatter_size"],
            edgecolors=CONFIG["STANDARD_DERIVE_CONFIG"]["plot"]["edgecolors"],
            linewidth=CONFIG["STANDARD_DERIVE_CONFIG"]["plot"]["linewidth"]
        )
        
        ax.set_xlabel('pH', fontsize=12, fontweight='bold', labelpad=3)
        ax.set_ylabel('PSS (%)', fontsize=12, fontweight='bold', labelpad=5)
        ax.set_zlabel('SOM (g/kg)', fontsize=12, fontweight='bold', labelpad=3)
        
        cbar = fig.colorbar(scatter, ax=ax, 
                           pad=CONFIG["STANDARD_DERIVE_CONFIG"]["plot"]["colorbar_pad"],
                           shrink=CONFIG["STANDARD_DERIVE_CONFIG"]["plot"]["colorbar_shrink"],
                           aspect=CONFIG["STANDARD_DERIVE_CONFIG"]["plot"]["colorbar_aspect"])
        cbar.set_label('Available Cd (mg/kg)', fontsize=12, fontweight='bold', labelpad=10)
        ax.view_init(elev=CONFIG["STANDARD_DERIVE_CONFIG"]["plot"]["view_elev"], 
                    azim=CONFIG["STANDARD_DERIVE_CONFIG"]["plot"]["view_azim"])
        
        # 展示结果
        st.subheader("🎯 The Results of standard derivation for ACd")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Median of ACd", f"{raw_median:.4f}")
        with col2:
            st.metric("R²", r2_log_scale)
        with col3:
            st.metric("The corrected ACd standard", corrected_median)
        
        with st.expander("📋 Parameter for derivation", expanded=True):
            var_info = pd.DataFrame({
                "Variables": ["TCd", "CEC", "SM", "pH", "PSS", "SOM"],
                "Value Type": ["Fixed Value (Original)", "Median (Original)", "Median (Original)", "Dynamic Range (Original)", "Dynamic Range (Original)", "Dynamic Range (Original)"],
                "Value/Range": [
                    f"{data_stats['TCd']['fixed_original']}（After Logarithmic Transformation：{grid_result['var_ranges']['TCd'][1]:.4f}）",
                    f"{data_stats['CEC']['median']}（After Logarithmic Transformation：{grid_result['var_ranges']['CEC'][1]:.4f}）",
                    f"{data_stats['SM']['median']}（After Logarithmic Transformation：{grid_result['var_ranges']['SM'][1]:.4f}）",
                    f"{data_stats['pH']['min']:.2f} ~ {data_stats['pH']['max']:.2f}",
                    f"{data_stats['PSS']['min']:.2f} ~ {data_stats['PSS']['max']:.2f}",
                    f"{data_stats['SOM']['min']:.2f} ~ {data_stats['SOM']['max']:.2f}"
                ]
            })
            st.dataframe(var_info, use_container_width=True)
        
        st.pyplot(fig, use_container_width=True)
        
        # 导出结果
        result_df = pd.DataFrame({
            "pH": pH_grid.flatten(),
            "PSS": PSS_grid.flatten(),
            "SOM": SOM_grid.flatten(),
            "Predictive values of ACd (original scale)": ACd_ori_flat
        })
        result_csv = result_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="💾 Downloading the 3D mesh ACd prediction results",
            data=result_csv,
            file_name="ACd standard derivation_3D mesh prediction results.csv",
            mime="text/csv"
        )
        
        return corrected_median
    except Exception as e:
        st.error(f"❌ Failure in deriving the ACd standard：{str(e)}")
        return None
