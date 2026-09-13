# 👁️ Q-FedSecure DR-XAI: Explainable AI for Diabetic Retinopathy Screening

**Clinical Edge Triage & Federated Explainable AI Dashboard for Rural Primary Health Centers (PHCs)**

---

## 📌 Overview
Q-FedSecure DR-XAI is an AI-driven, clinical-grade decision support platform designed to assist healthcare workers (ASHA/ANM) and tele-ophthalmologists in diagnosing and triaging Diabetic Retinopathy (DR) using retinal fundus photographs.

### Key Highlights
- **Multi-Class DR Grading**: Classifies images into No DR, Mild, Moderate, Severe, and Proliferative DR.
- **Explainable AI (XAI)**: Grad-CAM heatmap overlays highlighting microaneurysms, hemorrhages, and exudates.
- **Blood Vessel & Lesion Segmentation**: Morphological processing and segmentation masks for retinal analysis.
- **Automated Clinical Reporting**: PDF generation with diagnostic summaries, referral priority, and clinical recommendations.
- **Federated & Edge Ready**: Architecture built for secure multi-center collaboration without sharing raw patient data.

---

## 🛠️ Tech Stack
- **Framework**: Streamlit
- **Deep Learning**: TensorFlow / Keras, PennyLane
- **Computer Vision**: OpenCV, PIL, Scikit-Image
- **Reporting**: ReportLab
- **Federated Learning**: Flower (`flwr`)

---

## 🚀 Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/Nabisahebshaik/q-fedsecure-dr-xai.git
cd q-fedsecure-dr-xai
```

### 2. Set Up Virtual Environment
**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
streamlit run app.py
```

---

## 👥 Team Collaboration Workflow
- Always pull latest changes before coding: `git pull origin main`
- Create your feature branch: `git checkout -b feature/your-feature-name`
- Push your feature branch: `git push -u origin feature/your-feature-name`
- Submit a Pull Request on GitHub for review.
