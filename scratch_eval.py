import os
import cv2
import numpy as np
from src.preprocessing import assess_image_quality, apply_clahe_enhancement
from src.segmentation import segment_retinal_vessels, detect_lesion_candidates
from src.model import build_qcnet_model, predict_dr_grade
from src.xai import generate_gradcam_heatmap, overlay_heatmap_on_image, generate_clinical_explanation

image_path = r"C:/Users/shaik/.gemini/antigravity/brain/d50e5d08-3102-4e4f-8619-d6e6df292322/.user_uploaded/media_1787908548786.png"
img_raw = cv2.imread(image_path)
img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

print("=" * 65)
print("  Q-FedSecure DR-XAI: Clinical Diagnostic Analysis Report")
print("=" * 65)

# 1. IQA Assessment
is_valid, metrics = assess_image_quality(img)
print("\n[1] Real-Time Image Quality Assessment (IQA)")
print(f"  • Quality Status:       {'PASS (Gradeable)' if is_valid else 'REJECT'}")
print(f"  • Laplacian Sharpness:  {metrics['laplacian_variance']:.2f}")
print(f"  • Mean Luminance:       {metrics['mean_intensity']:.2f}")
print(f"  • Retinal FOV Coverage: {metrics['fov_coverage_ratio']*100:.1f}%")

# 2. Preprocessing & Biomarker Detection
clahe_img = apply_clahe_enhancement(img)
v_mask, v_overlay, v_density = segment_retinal_vessels(img)
lesions = detect_lesion_candidates(img)
print("\n[2] Retinal Biomarker & Pathology Quantification")
print(f"  • Vascular Tree Density:          {v_density:.2f}%")
print(f"  • Red Lesions / Hemorrhages:      {lesions['red_lesions_count']}")
print(f"  • Hard Exudates / Lipid Leaks:    {lesions['bright_lesions_count']}")

# 3. Model Inference (DenseNet201 + 4-Qubit PennyLane VQC)
model = build_qcnet_model(num_classes=5, weights_path="weights/qcnet_weights.weights.h5")
grade_idx, grade_name, conf, probs = predict_dr_grade(model, clahe_img, lesion_stats=lesions, vessel_density=v_density)

print("\n[3] QCNET Quantum-Classical Deep Learning Prediction")
print(f"  • Diagnostic Verdict:    {grade_name}")
print(f"  • Model Confidence:      {conf*100:.2f}%")
print("\n  ICDR 5-Class Probability Distribution:")
grade_labels = [
    "Grade 0: No DR (Normal)",
    "Grade 1: Mild NPDR",
    "Grade 2: Moderate NPDR",
    "Grade 3: Severe NPDR",
    "Grade 4: Proliferative DR (PDR)"
]
for g, p in zip(grade_labels, probs):
    bar = "█" * int(p * 25)
    print(f"    {g:<34}: {p*100:5.2f}% | {bar}")

# 4. Explainable AI & Clinical Action
heatmap = generate_gradcam_heatmap(model, clahe_img, target_class_idx=grade_idx)
explanation = generate_clinical_explanation(grade_idx, conf, v_density, lesions)

print("\n[4] Clinical Triage & Risk Stratification")
print(f"  • Triage Urgency Level:  {explanation['urgency_level'].upper()} URGENCY")
print(f"  • Pathological Findings: {explanation['findings']}")
print(f"  • Actionable Directive:  {explanation['recommendation']}")
print("=" * 65)
