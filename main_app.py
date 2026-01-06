import streamlit as st
from config.settings import CONFIG
from utils.utils import load_model
from utils.features import batch_analyze_files, derive_acd_standard

def main():
    # 页面配置
    st.set_page_config(
        page_title="Standard derivation of ACd",
        page_icon="📊",
        layout="wide"
    )
    st.title("📊 The batch analysis and standard derivation of ACd")
    st.divider()
    
    # 加载模型
    with st.spinner("🔧 加载ACd预测模型..."):
        try:
            model = load_model()
            st.success("✅ Successfully loading prediction model (XGBoost)")
        except Exception as e:
            st.error(f"❌ 模型加载失败：{str(e)}")
            return
    
    # 批量文件上传与分析
    st.subheader("🔹 Step1: Batch upload of sample data")
    uploaded_files = st.file_uploader(
        "File format: CSV or XLSX",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help=f"文件需包含列：{', '.join(CONFIG['FEATURE_COLS'] + [CONFIG['TARGET_COL']])}"
    )
    
    data_stats = None
    r2_log_scale = None
    
    if uploaded_files:
        st.divider()
        st.subheader("🔹 Step2: Analysis results of batch data")
        data_stats, r2_log_scale = batch_analyze_files(uploaded_files, model)
    
    # ACd标准推导
    st.divider()
    st.subheader("🔹 Step 3: Derivation of the standard for ACd (3D prediction)")
    if data_stats is None or r2_log_scale is None:
        st.info("💡Please complete the batch data upload and analysis of steps 1-2 firstly. The system will then derive the ACd standard based on the variable range of the uploaded data.")
    else:
        with st.expander("⚙️ Variable range can be adjusted (optional)", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                data_stats["pH"]["min"] = st.number_input("pH_min", value=float(data_stats["pH"]["min"]), format="%.2f")
                data_stats["pH"]["max"] = st.number_input("pH_max", value=float(data_stats["pH"]["max"]), format="%.2f")
            with col2:
                data_stats["PSS"]["min"] = st.number_input("PSS_min", value=float(data_stats["PSS"]["min"]), format="%.2f")
                data_stats["PSS"]["max"] = st.number_input("PSS_max", value=float(data_stats["PSS"]["max"]), format="%.2f")
            with col3:
                data_stats["SOM"]["min"] = st.number_input("SOM_min", value=float(data_stats["SOM"]["min"]), format="%.2f")
                data_stats["SOM"]["max"] = st.number_input("SOM_max", value=float(data_stats["SOM"]["max"]), format="%.2f")
        
        if st.button("🚀 Start to derive the ACd standard", type="primary"):
            with st.spinner("The 3D mesh is being generated and the ACd standard is being derived...."):
                derive_acd_standard(model, data_stats, r2_log_scale)

if __name__ == "__main__":
    main()


#https://standard-derivation-acd-soil-for-gm-sc.streamlit.app/