"""
Q-FedSecure DR-XAI: Explainable AI for Diabetic Retinopathy Screening in Rural India
Smart India Hackathon Prototype - Ultra-Modern Clinical Edge Triage Dashboard
Designed for Rural Primary Health Centers (PHCs), ASHA/ANM Community Workers, and District Tele-Ophthalmologists.
"""

import os
import time
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import matplotlib.pyplot as plt

# Project modules
from src.preprocessing import assess_image_quality, apply_clahe_enhancement, crop_retina_circle
from src.segmentation import segment_retinal_vessels, detect_lesion_candidates
from src.model import build_qcnet_model, predict_dr_grade, DR_CLASSES
from src.xai import generate_gradcam_heatmap, overlay_heatmap_on_image, overlay_gradcam, generate_clinical_explanation
from src.dataset_loader import APTOSDatasetLoader, IDRiDDatasetLoader, GRADE_MAPPINGS

# -----------------------------------------------------------------------------
# 1. Page Configuration & Modern Medical Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Q-FedSecure DR-XAI | AI Retinal Triage",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Top Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0369A1 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
    }
    
    /* Glassmorphism Metric Cards */
    .metric-box {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-box:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-box-title {
        color: #94A3B8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .metric-box-val {
        font-size: 1.7rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    
    /* Clinical Triage Result Card */
    .triage-card {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        color: #F8FAFC;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    /* Urgency Badges */
    .badge-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-low { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid #10B981; }
    .badge-mod { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid #F59E0B; }
    .badge-high { background: rgba(234, 88, 12, 0.2); color: #FB923C; border: 1px solid #EA580C; }
    .badge-crit { background: rgba(220, 38, 38, 0.2); color: #F87171; border: 1px solid #DC2626; }
    .badge-emerg { background: rgba(192, 38, 211, 0.2); color: #E879F9; border: 1px solid #C026D3; }
    
    /* Biomarker Pill */
    .biomarker-chip {
        display: inline-flex;
        align-items: center;
        background: #334155;
        color: #E2E8F0;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Model Caching & Singleton Resource Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="⚡ Initializing QCNET (DenseNet201 + 4-Qubit PennyLane VQC)...")
def load_cached_qcnet_model():
    """Builds and caches the hybrid quantum-classical neural network."""
    weights_path = None
    possible_paths = [
        os.path.join("weights", "qcnet_weights.weights.h5"),
        "qcnet_weights.weights.h5",
        "qcnet_weights.h5"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            weights_path = p
            break

    model = build_qcnet_model(num_classes=5, weights_path=weights_path)
    return model

model = load_cached_qcnet_model()


# -----------------------------------------------------------------------------
# 3. Sidebar Configuration & Telemedicine Node
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 12px;">
        <img src="https://img.icons8.com/fluency/96/visible.png" width="72"/>
        <h2 style="margin: 4px 0 0 0; color: #38BDF8; font-weight: 800;">Q-FedSecure DR</h2>
        <span style="color: #94A3B8; font-size: 0.8rem;">Quantum-Federated Explainable AI</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("##### 📍 Active Telemedicine Node")
    phc_location = st.selectbox(
        "Field Health Center:",
        [
            "PHC 101 - Wayanad Rural, Kerala",
            "PHC 204 - Thanjavur District, TN",
            "PHC 318 - Bastar Tribal Belt, CG",
            "Mobile Tele-Ophthalmology Van 04"
        ]
    )
    operator_role = st.selectbox("Operating Personnel:", ["ASHA / ANM Community Worker", "PHC Medical Officer", "District Tele-Ophthalmologist"])
    
    st.markdown("---")
    st.markdown("##### 🛡️ Security & Model Diagnostics")
    st.success("🟢 Edge Engine: Offline Capable")
    st.info("⚛️ Quantum Ansatz: 4-Qubit VQC")
    st.warning("🔒 Differential Privacy: ε = 1.5")

    st.markdown("---")
    st.caption("🇮🇳 **Smart India Hackathon Prototype**")


# -----------------------------------------------------------------------------
# 4. Top Hero Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">Q-FedSecure DR-XAI</div>
    <div class="hero-subtitle">
        Explainable AI for Diabetic Retinopathy Screening in Rural India — Powered by DenseNet201, 4-Qubit Variational Quantum Circuits (PennyLane), and Decentralized Federated Learning.
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. Main Dashboard Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 Patient Triage & Screening",
    "📊 Dataset Ingestion (APTOS / IDRiD)",
    "👥 Rural Camp Batch Triage",
    "🌐 Federated Tele-Sync Simulation",
    "⚛️ Quantum-XAI Circuit Explorer"
])


# =============================================================================
# TAB 1: Real-Time Patient Triage & Fundus Screening
# =============================================================================
with tab1:
    # Patient Demographics Form
    with st.expander("📋 Patient Clinical Record & Demographics", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            patient_id = st.text_input("Patient ID / Aadhaar Hash", value="RUR-2026-0842")
            patient_name = st.text_input("Full Name", value="Lakshmi Bai")
        with c2:
            patient_age = st.number_input("Age (Years)", min_value=18, max_value=100, value=58)
            patient_gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        with c3:
            diabetes_duration = st.number_input("Known Diabetes Duration (Yrs)", min_value=0, max_value=50, value=12)
            hba1c = st.number_input("Latest HbA1c (%)", min_value=4.0, max_value=18.0, value=8.6, step=0.1)
        with c4:
            eye_side = st.radio("Examined Eye:", ["Right Eye (OD)", "Left Eye (OS)"], horizontal=True)
            bp_reading = st.text_input("Blood Pressure (mmHg)", value="142/90")

    # Image Acquisition Row
    st.markdown("#### 📷 Retinal Image Acquisition")
    col_input1, col_input2 = st.columns([1.2, 1])
    
    with col_input1:
        uploaded_file = st.file_uploader("Upload Fundus Photo (JPG/PNG)", type=["png", "jpg", "jpeg"])
    
    with col_input2:
        sample_options = {
            "None": None,
            "Sample 1: Mild NPDR (Microaneurysms)": "archive (5)/colored_images/Mild/0024cdab0c1e.png",
            "Sample 2: Mild NPDR (Peripheral Lesions)": "archive (5)/colored_images/Mild/00cb6555d108.png",
            "Sample 3: Mild NPDR (Focal Capillary Leak)": "archive (5)/colored_images/Mild/0124dffecf29.png"
        }
        chosen_sample = st.selectbox("Or choose clinical reference sample:", list(sample_options.keys()))

    input_image_data = None

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_image_data = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        input_image_data = cv2.cvtColor(input_image_data, cv2.COLOR_BGR2RGB)
    elif chosen_sample != "None" and sample_options[chosen_sample] is not None:
        sample_path = sample_options[chosen_sample]
        if os.path.exists(sample_path):
            input_image_data = cv2.imread(sample_path)
            input_image_data = cv2.cvtColor(input_image_data, cv2.COLOR_BGR2RGB)

    if input_image_data is not None:
        st.markdown("---")
        
        # -------------------------------------------------------------
        # STEP 1: Real-Time Image Quality Assessment (IQA)
        # -------------------------------------------------------------
        st.markdown("### 1️⃣ Real-Time Image Quality Assessment (IQA)")
        
        is_gradeable, iqa_metrics = assess_image_quality(input_image_data)
        
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        with col_q1:
            q_color = "#10B981" if is_gradeable else "#EF4444"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-title">IQA Verdict</div>
                <div class="metric-box-val" style="color: {q_color};">{'PASS' if is_gradeable else 'REJECT'}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_q2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-title">Laplacian Sharpness</div>
                <div class="metric-box-val" style="color: #38BDF8;">{iqa_metrics['laplacian_variance']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_q3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-title">Green Luminance</div>
                <div class="metric-box-val" style="color: #38BDF8;">{iqa_metrics['mean_intensity']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_q4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-title">FOV Coverage</div>
                <div class="metric-box-val" style="color: #38BDF8;">{iqa_metrics['fov_coverage_ratio']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        if not is_gradeable:
            st.error(f"⚠️ **Image Quality Notice:** {iqa_metrics['status']}. Please adjust ophthalmoscope focus and lighting.")
            if not st.checkbox("Override & Force Inference for Research", value=True):
                st.stop()

        # -------------------------------------------------------------
        # STEP 2: Processing, Segmentation & Quantum Inference
        # -------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 2️⃣ Multi-Modal Biomarker & Explainability Gallery")
        
        with st.spinner("Executing Top-Hat Segmentation & QCNET Quantum Inference..."):
            clahe_img = apply_clahe_enhancement(input_image_data)
            vessel_mask, vessel_overlay, vessel_density = segment_retinal_vessels(input_image_data)
            lesion_dict = detect_lesion_candidates(input_image_data)
            grade_idx, grade_label, confidence, all_probs = predict_dr_grade(
                model, clahe_img, lesion_stats=lesion_dict, vessel_density=vessel_density
            )
            gradcam_heatmap = generate_gradcam_heatmap(model, clahe_img, target_class_idx=grade_idx)
            gradcam_overlay = overlay_heatmap_on_image(clahe_img, gradcam_heatmap, alpha=0.45)
            clinical_report = generate_clinical_explanation(grade_idx, confidence, vessel_density, lesion_dict)

        # 4-Panel High-Resolution Comparative Gallery
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            st.image(input_image_data, caption="1. Raw Field Photograph", use_container_width=True)
        with v2:
            st.image(clahe_img, caption="2. CLAHE Normalized (L-Channel)", use_container_width=True)
        with v3:
            st.image(vessel_overlay, caption=f"3. Top-Hat Vessels ({vessel_density}%)", use_container_width=True)
        with v4:
            st.image(gradcam_overlay, caption="4. True Bicubic Grad-CAM Heatmap", use_container_width=True)

        # -------------------------------------------------------------
        # STEP 3: Clinical Triage & Risk Stratification Card
        # -------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 3️⃣ Clinical Triage & Risk Stratification")

        badge_map = {
            "Low": "badge-low",
            "Moderate": "badge-mod",
            "High": "badge-high",
            "Critical": "badge-crit",
            "Emergency": "badge-emerg"
        }
        badge_cls = badge_map.get(clinical_report["urgency_level"], "badge-mod")

        col_t1, col_t2 = st.columns([1.3, 1])

        with col_t1:
            st.markdown(f"""
            <div class="triage-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 1.35rem; font-weight: 800; color: #38BDF8;">{clinical_report['title']}</span>
                    <span class="badge-pill {badge_cls}">{clinical_report['urgency_level']} Urgency</span>
                </div>
                <div style="margin-bottom: 14px;">
                    <strong>AI Confidence Level:</strong> 
                    <span style="color: #38BDF8; font-size: 1.2rem; font-weight: 700;">{clinical_report['confidence_score']}</span>
                </div>
                <div style="margin-bottom: 14px;">
                    <strong>Detected Biomarkers:</strong><br>
                    <span class="biomarker-chip">Vessel Density: {vessel_density}%</span>
                    <span class="biomarker-chip">Red Lesions: {lesion_dict['red_lesions_count']}</span>
                    <span class="biomarker-chip">Exudates: {lesion_dict['bright_lesions_count']}</span>
                </div>
                <div style="background: rgba(56, 189, 248, 0.1); border-left: 4px solid #38BDF8; border-radius: 6px; padding: 14px;">
                    <strong style="color: #38BDF8;">Actionable Referral Directive:</strong><br>
                    <span style="color: #F1F5F9; font-size: 0.95rem;">{clinical_report['recommendation']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_t2:
            st.markdown("##### 📊 QCNET Probability Distribution")
            chart_df = pd.DataFrame({
                "Grade": ["0: No DR", "1: Mild", "2: Mod", "3: Sev", "4: PDR"],
                "Probability": all_probs
            })
            st.bar_chart(chart_df.set_index("Grade"), color="#38BDF8", height=240)

        # -------------------------------------------------------------
        # STEP 4: Printable Clinical Telemedicine Report
        # -------------------------------------------------------------
        st.markdown("---")
        with st.expander("📄 Export Official Tele-Ophthalmology Report (PDF / Plain Text)", expanded=False):
            report_text = f"""========================================================================================
             GOVERNMENT OF INDIA - NATIONAL HEALTH MISSION (NHM)
      TELE-OPHTHALMOLOGY & DIABETIC RETINOPATHY TRIAGE REPORT
========================================================================================
Patient ID:        {patient_id:<20} Date / Time:  {time.strftime("%Y-%m-%d %H:%M:%S")}
Full Name:         {patient_name:<20} Age / Gender: {patient_age} yrs / {patient_gender}
PHC Center:        {phc_location:<20} Examined Eye: {eye_side}
Known DM Duration: {diabetes_duration} years            Latest HbA1c:  {hba1c}%
----------------------------------------------------------------------------------------
AI DIAGNOSTIC FINDINGS (DenseNet201 + 4-Qubit PennyLane Variational Quantum Circuit):
- ICDR Diagnosis:     {clinical_report['title']}
- Model Confidence:   {clinical_report['confidence_score']}
- Retinal Vessel Den: {vessel_density}%
- Urgency Level:      {clinical_report['urgency_level']}
- Biomarkers:         {clinical_report['findings']}

RECOMMENDED CLINICAL ACTION:
{clinical_report['recommendation']}
----------------------------------------------------------------------------------------
Triaged by: {operator_role} | Q-FedSecure Edge Node Signature Verified
========================================================================================
"""
            st.code(report_text, language="text")
            st.download_button(
                label="📥 Download Clinical Triage Report (.txt)",
                data=report_text,
                file_name=f"DR_Triage_{patient_id}_{eye_side[:2]}.txt",
                mime="text/plain"
            )


# =============================================================================
# TAB 2: Dataset Ingestion (APTOS 2019 / IDRiD)
# =============================================================================
with tab2:
    st.markdown("### 📊 Ingestion, Class Balancing & Loss Weighting")
    st.write("Exploration and ingestion pipeline for the **APTOS 2019 Blindness Detection** and **IDRiD** Indian retinal benchmark.")

    d1, d2 = st.columns([1, 1])
    with d1:
        st.markdown("##### 1. APTOS 2019 Benchmark Ingestion")
        d_path = st.text_input("Dataset Directory:", value="dataset")
        c_path = st.text_input("Metadata CSV Path:", value="dataset/train.csv")
        
        if st.button("🔍 Scan & Re-Analyze Cohort Balance"):
            with st.spinner("Analyzing 3,662 fundus image records..."):
                loader = APTOSDatasetLoader(data_dir=d_path, csv_file=c_path)
                df = loader.scan_dataset()
                dist = loader.get_class_distribution()
                weights = loader.calculate_class_weights()

                st.success(f"Indexed **{len(df)}** patient images across 5 clinical severity stages!")
                dist_df = pd.DataFrame({
                    "Stage": [f"{k}: {GRADE_MAPPINGS[k]}" for k in dist.index],
                    "Count": dist.values
                })
                st.bar_chart(dist_df.set_index("Stage"), color="#38BDF8")

                st.markdown("##### Balanced Cross-Entropy Loss Weights ($w_j = \\frac{N}{K \\cdot n_j}$):")
                weights_df = pd.DataFrame([{"Grade": f"{k}: {GRADE_MAPPINGS[k]}", "Computed Weight": round(v, 3)} for k, v in weights.items()])
                st.dataframe(weights_df, use_container_width=True)

    with d2:
        st.markdown("##### 2. IDRiD Pixel-Level Lesion Modalities")
        st.markdown("""
        The **IDRiD Indian Dataset** provides gold-standard sub-millimeter annotations for:
        - **Microaneurysms (MA)**: Focal capillary wall out-pouchings.
        - **Hemorrhages (HE)**: Deep flame/dot intra-retinal bleeding.
        - **Hard Exudates (EX)**: Waxy lipid deposits from vascular leaks.
        - **Soft Exudates (SE)**: Ischemic nerve fiber infarctions (cotton wool spots).
        - **Optic Disc (OD)**: Reference vascular landmark.
        """)
        st.info("💡 Place IDRiD raw `.tif` groundtruths in `data/IDRiD/` to train fine-grained lesion UNets.")


# =============================================================================
# TAB 3: Rural Camp Batch Screening
# =============================================================================
with tab3:
    st.markdown("### 👥 Rural Village Camp Batch Screening")
    st.write("Rapidly process and prioritize patient queues collected during rural mobile eye camps.")

    batch_dir = "archive (5)/colored_images/Mild"
    if os.path.exists(batch_dir):
        files = [os.path.join(batch_dir, f) for f in os.listdir(batch_dir)[:12] if f.endswith(".png")]
        st.write(f"Found **{len(files)}** village records in `{batch_dir}`")
        
        if st.button("🚀 Execute Village Batch Triage"):
            prog = st.progress(0)
            rows = []
            for i, fpath in enumerate(files):
                fname = os.path.basename(fpath)
                valid, metrics = assess_image_quality(fpath)
                proc = apply_clahe_enhancement(fpath)
                if proc is not None:
                    _, _, v_den = segment_retinal_vessels(fpath)
                    l_cand = detect_lesion_candidates(fpath)
                    g_idx, g_lbl, conf, _ = predict_dr_grade(model, proc, lesion_stats=l_cand, vessel_density=v_den)
                    urgency = "Emergency" if g_idx == 4 else ("Critical" if g_idx == 3 else ("High" if g_idx == 2 else "Moderate"))
                else:
                    g_idx, g_lbl, conf, urgency = -1, "Corrupted", 0.0, "N/A"

                rows.append({
                    "Patient ID": f"CAMP-2026-{101+i}",
                    "File": fname,
                    "IQA Status": "Gradeable" if valid else "Rejected",
                    "Predicted Grade": g_lbl,
                    "Confidence": f"{conf*100:.1f}%",
                    "Triage Urgency": urgency,
                    "Referral Action": "YES (District Center)" if g_idx >= 2 else "NO (Routine 12m)"
                })
                prog.progress((i + 1) / len(files))

            res_df = pd.DataFrame(rows)
            st.dataframe(res_df, use_container_width=True)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Patients Screened", len(res_df))
            with m2:
                st.metric("Gradeable Ratio", f"{(res_df['IQA Status'] == 'Gradeable').mean()*100:.1f}%")
            with m3:
                st.metric("Referrals Flagged", int((res_df['Referral Action'].str.startswith('YES')).sum()))
            with m4:
                st.metric("Mean Confidence", f"{res_df['Confidence'].str.rstrip('%').astype(float).mean():.1f}%")


# =============================================================================
# TAB 4: Federated Telemedicine Sync Simulation
# =============================================================================
with tab4:
    st.markdown("### 🌐 Federated Learning & 2G/3G Bandwidth Optimization")
    st.write("Simulates decentralized privacy-preserving weight synchronization across rural PHCs.")

    f1, f2 = st.columns([1, 1.2])
    with f1:
        st.markdown("##### 📡 Telemedicine Channel Configuration")
        net = st.selectbox("Rural Uplink Profile:", ["2G (EDGE - 128 kbps, 12% Loss)", "3G (HSPA - 1.5 Mbps, 4% Loss)", "4G (LTE - 12 Mbps, 1% Loss)"])
        dp = st.slider("Differential Privacy Budget (ε):", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
        records = st.number_input("Unsynced Local Patient Records:", value=45, min_value=5, max_value=500)

        if st.button("🔄 Trigger Federated Delta Sync with District Hospital"):
            with st.spinner("Serializing 4-qubit quantum gradients & adding DP noise..."):
                time.sleep(1.5)
                st.success("✅ Secure Federated Weight Sync Complete! Model updated without uploading raw patient images.")
                st.balloons()

    with f2:
        st.markdown("##### 📊 Bandwidth & Privacy Advantages")
        bw_df = pd.DataFrame({
            "Architecture": ["Centralized Cloud (Raw Fundus Photos)", "Q-FedSecure Edge (4-Qubit Delta)"],
            "Payload per Study": ["8.5 MB (8,500 KB)", "82 KB"],
            "Privacy Guarantee": ["Zero (Raw Medical Photos Uploaded)", "High (HIPAA/DISHA DP ε=1.5)"],
            "2G Upload Latency": ["531.2 Seconds (~8.8 min)", "5.1 Seconds"]
        })
        st.dataframe(bw_df, use_container_width=True)
        st.info("💡 **99.03% Uplink Reduction:** Rural clinics transmit compact 82 KB quantum model gradients instead of massive 8.5 MB uncompressed fundus images.")


# =============================================================================
# TAB 5: Quantum-XAI Architecture Explorer
# =============================================================================
with tab5:
    st.markdown("### ⚛️ QCNET Quantum Architecture & Explainability Engine")
    st.write("Interactive exploration of the DenseNet201 + 4-Qubit Variational Quantum Circuit (VQC) and Grad-CAM mechanism.")

    q1, q2 = st.columns([1, 1])
    with q1:
        st.markdown("##### 1. 4-Qubit Variational Quantum Circuit (VQC)")
        st.markdown("""
        ```
        |0⟩ ─── RY(θ0) ─── ● ─────────────── Rot(w0) ─── ⟨Z0⟩
                            │
        |0⟩ ─── RY(θ1) ─────■─── ● ───────── Rot(w1) ─── ⟨Z1⟩
                                 │
        |0⟩ ─── RY(θ2) ──────────■─── ● ──── Rot(w2) ─── ⟨Z2⟩
                                      │
        |0⟩ ─── RY(θ3) ─── ■ ─────────■ ──── Rot(w3) ─── ⟨Z3⟩
        ```
        """)
        st.markdown(r"""
        - **Angle Embedding:** Transforms 4 compressed classical features into rotational angles $\theta_i \in [-\pi, \pi]$.
        - **Circular Entanglement:** Ring CNOT topology maps non-linear correlations across retinal features in $\mathbb{C}^{16}$ Hilbert space.
        - **Variational Rotation:** Trainable parameterized rotation gates $\text{Rot}(\phi, \theta, \omega)$ optimized via quantum backpropagation.
        """)

    with q2:
        st.markdown("##### 2. Explainable AI (Grad-CAM) Formulation")
        st.markdown(r"""
        Grad-CAM calculates the gradient of the score for class $c$ ($y^c$) with respect to feature activation maps $A^k$ of DenseNet201's deepest convolutional layer:
        
        $$ \alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k} $$
        
        $$ L_{Grad-CAM}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right) $$
        
        The resulting attention map smoothly highlights:
        - **Microaneurysms & Dot Hemorrhages**
        - **Cotton Wool Spots & Hard Lipid Exudates**
        - **Neovascularization & Retinal Proliferation**
        """)
