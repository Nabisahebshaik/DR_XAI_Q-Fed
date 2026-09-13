# Q-FedSecure DR-XAI: A Quantum-Classical Federated Learning Architecture with Explainable AI and Dual-Stream Biomarker Fusion for Diabetic Retinopathy Screening in Resource-Constrained Healthcare

**Target Conference:** International Conference on Artificial Intelligence and Networking (ICAIN-2026)  
**Organized by:** BITS Pilani, Dubai Campus in association with IIIT Allahabad  
**Special Session 3:** *AI and Data Analytics for Intelligent Decision Systems in Smart Healthcare*  
**Publication Track:** Springer Lecture Notes in Networks and Systems (LNNS) (SCOPUS & WoS Indexed)  

---

### Authors
**Nabisaheb Shaik$^{1}$, Lead AI Architect & SIH Research Team**  
$^{1}$Department of Computer Science & Engineering / Biomedical Informatics  
*(Correspondence Email: shaiknabisaheb187@gmail.com)*

---

### Abstract
Accurate, privacy-preserving, and bandwidth-efficient automated screening of Diabetic Retinopathy (DR) remains a vital challenge in developing rural healthcare systems. Deep convolutional neural networks and vision transformers suffer from heavy parameter footprints, susceptibility to high-frequency sensor noise on low-cost ophthalmoscopes, and vulnerability to catastrophic false-negative misclassifications when softmax distributions are close to uniform. Furthermore, transmitting raw high-resolution retinal fundus photographs (8.5 MB) over rural 2G/3G communication links severely violates national healthcare privacy laws (e.g., DISHA, HIPAA) and induces severe network latency. To address these multi-faceted challenges, we propose **Q-FedSecure DR-XAI**, a novel decentralized quantum-classical medical intelligence framework. 

Our model extracts hierarchical convolutional bottleneck representations using DenseNet201 and embeds them into a 4-qubit **Variational Quantum Circuit (VQC)** operating in a 16-dimensional Hilbert state space ($\mathbb{C}^{16}$), capturing non-linear feature entanglements with only 12 trainable quantum parameters. To guarantee mathematical privacy and eliminate transmission bottlenecks, we deploy decentralized **Federated Learning with Differential Privacy ($\epsilon = 1.5$)**, transmitting only compact 82 KB quantum gradient deltas to achieve a **99.03% uplink bandwidth reduction**. Furthermore, to eliminate "black-box" normal classifications on diseased retinas, we introduce a **Dual-Stream Bayesian Clinical Biomarker Decision Fusion Layer** that merges quantum predictions with multi-scale morphological Top-Hat vascular tree density and isoperimetric circularity-filtered lesion candidates calibrated against the International Clinical Diabetic Retinopathy (ICDR) scale. Explainability is established via true bicubic Gradient-Weighted Class Activation Mapping (Grad-CAM). Experimental evaluations on the benchmark APTOS 2019 ($N=3,662$) and IDRiD datasets, alongside $M/M/c$ district tele-queuing simulations, demonstrate that Q-FedSecure DR-XAI achieves superior diagnostic robustness, zero false-negative normal classifications, and optimal edge deployability for rural telemedicine.

**Index Terms—** Diabetic Retinopathy, Quantum Machine Learning, Variational Quantum Circuit (VQC), Federated Learning, Differential Privacy, Explainable AI (Grad-CAM), Smart Healthcare, Tele-Ophthalmology.

---

## I. INTRODUCTION

Diabetic Retinopathy (DR) is a microvascular complication of diabetes mellitus and constitutes the leading cause of preventable visual impairment and irreversible blindness among working-age adults worldwide. In developing nations such as India, over 77 million individuals live with diabetes, yet early epidemiological screening is severely constrained by an acute shortage of trained ophthalmologists in rural Primary Health Centers (PHCs). Community healthcare workers (e.g., ASHAs and ANMs) equipped with low-cost, non-mydriatic portable fundus cameras represent the primary frontline defense. However, deploying classical cloud-based artificial intelligence (AI) models in these settings introduces four fundamental bottlenecks:

1. **Bandwidth Bottlenecks in Rural Networks:** Standard fundus photographs captured at clinical resolutions range between 5 MB and 12 MB per eye. In remote rural districts operating under 2G (EDGE) or congested 3G networks, transmitting raw volumetric images to centralized cloud servers results in transmission latencies exceeding 8 to 10 minutes per patient study, leading to frequent session dropouts.
2. **Healthcare Data Privacy & Regulatory Compliance:** Centralized aggregation of identifiable retinal photographs violates patient data privacy frameworks, including the Digital Information Security in Healthcare Act (DISHA) and HIPAA. A decentralized edge architecture is mandatory.
3. **Black-Box Softmax Calibration Failures:** Deep neural networks trained on imbalanced datasets often produce near-uniform softmax probability distributions ($\approx 20\%$ per class) on subtle pathological boundaries. In standard argmax decision heads, a minor $0.8\%$ statistical deviation can cause an eye containing 14 active intra-retinal hemorrhages to be misclassified as "Grade 0: Normal Retina"—a catastrophic diagnostic error.
4. **Sensor Noise & High-Parameter Overfitting:** Purely classical CNNs ($\sim 25\text{M}+$ parameters) overfit to high-end hospital benchtop fundus cameras and exhibit severe domain degradation when applied to noisy, illumination-varying portable devices.

To overcome these challenges, we introduce **Q-FedSecure DR-XAI**, a hybrid quantum-classical federated framework. The main contributions of this work are summarized as follows:

* **Contribution 1 (Quantum-Classical Architecture - QCNET):** We propose a hybrid architecture combining a DenseNet201 deep feature extractor with a 4-qubit Parameterized Variational Quantum Circuit (VQC) in PennyLane. By leveraging quantum state superposition and circular $CNOT$ entanglement in a 16-dimensional Hilbert state space ($\mathbb{C}^{16}$), the model captures complex cross-quadrant retinal correlations with minimal parameter overhead.
* **Contribution 2 (Privacy-Preserving Federated Sync with 99.03% Bandwidth Reduction):** We formulate a decentralized Federated Learning pipeline with $(\epsilon = 1.5, \delta = 10^{-5})$-Differential Privacy. Transmitting only 82 KB compressed quantum weight gradient updates achieves a 99.03% bandwidth reduction over raw image streaming.
* **Contribution 3 (Dual-Stream Bayesian Clinical Decision Fusion):** We design a closed-loop decision layer that mathematically fuses deep quantum logits with explicit morphological retinal biomarkers (microaneurysms, hemorrhages, hard lipid exudates, and vascular density), enforcing ICDR clinical guidelines and eliminating false-negative normal classifications.
* **Contribution 4 (Multi-Scale Morphological Top-Hat & Isoperimetric Filtering):** We develop an automated vessel extraction algorithm using dual-scale Top-Hat transforms and isoperimetric circularity filtering ($C = \frac{4\pi A}{P^2}$) that isolates continuous tubular vessels while rejecting circular dot lesions.
* **Contribution 5 (Transparent Auditability via Bicubic Grad-CAM):** We implement true gradient-weighted activation mapping with bicubic interpolation (`cv2.INTER_CUBIC`), automatically synthesizing National Health Mission (NHM)-compliant triage reports.

---

## II. RELATED WORK

### A. Deep Learning in Diabetic Retinopathy Detection
Automated DR grading has seen significant progress through deep convolutional networks such as ResNet, DenseNet, and Vision Transformers (ViTs). DenseNet architectures in particular mitigate vanishing gradients and maximize feature reuse via dense connectivity. However, classical deep networks require high compute budgets, lack intrinsic physical interpretability, and suffer from high vulnerability to label noise on edge devices.

### B. Quantum Machine Learning (QML) in Biomedical Imaging
Parameterized Quantum Circuits (PQCs) and Variational Quantum Circuits (VQCs) have emerged as powerful paradigms for processing high-dimensional data. By encoding classical vectors into quantum amplitudes or rotation angles, quantum circuits exploit the exponential dimensionality of Hilbert space ($\mathcal{H} = \mathbb{C}^{2^n}$) to model non-linear interactions that are computationally intractable for classical networks with equivalent parameter counts.

### C. Federated Learning & Differential Privacy in Smart Healthcare
Federated Learning (FL) enables distributed client nodes (e.g., rural PHCs) to collaboratively train a shared global model while retaining all raw medical images locally. To prevent gradient inversion and reconstruction attacks, $(\epsilon, \delta)$-Differential Privacy injects calibrated Laplacian or Gaussian noise into client weight deltas prior to aggregation.

---

## III. PROPOSED METHODOLOGY

```
+-----------------------------------------------------------------------------------+
|                           Q-FedSecure DR-XAI Framework                            |
+-----------------------------------------------------------------------------------+
                                          |
                                   [Raw Fundus Image]
                                          |
                        +-----------------+-----------------+
                        |                                   |
                        v                                   v
             [Stream 1: Deep Quantum]             [Stream 2: Biomarkers]
                        |                                   |
               DenseNet201 Backbone               LAB-CLAHE Enhancement
                        |                                   |
             4D Bottleneck (tanh * pi)            Multi-Scale Top-Hat Transform
                        |                                   |
              4-Qubit VQC (PennyLane)             Isoperimetric Contour Filter
             - Angle Embedding: RY(theta)        - Vessel Density (rho)
             - Ring Entanglement: CNOT           - Red Lesions (Microaneurysms/HE)
             - Variational: Rot(phi,theta,w)     - Hard Exudates (Lipid Leaks)
             - Readout: <Z_0>,...,<Z_3>                     |
                        |                                   |
             Softmax Logits P_QCNET                         |
                        |                                   |
                        +-----------------+-----------------+
                                          |
                                          v
                     [Dual-Stream Bayesian Decision Fusion]
                      P(Grade | Image, Biomarkers) ~ P_QCNET * P_ICDR
                                          |
                                          v
                      [Bicubic Grad-CAM Attention Heatmap]
                                          |
                                          v
                     [Decentralized FL Client Update (DP-SGD)]
```

### A. 4-Qubit Variational Quantum Circuit (VQC) Formulation
Let $\mathbf{x} = [x_0, x_1, x_2, x_3]^T \in \mathbb{R}^4$ denote the classical bottleneck feature vector extracted from DenseNet201 after Global Average Pooling, Batch Normalization, and $\tanh$ activation scaled by $\pi$:
$$x_i = \pi \cdot \tanh(\mathbf{w}_b^T \mathbf{f}_{\text{conv}} + b_b), \quad x_i \in [-\pi, \pi]$$

The quantum circuit operates across $N = 4$ qubits initialized in the ground state $|0\rangle^{\otimes 4}$.

1. **State Preparation (Angle Embedding):**
   $$|\psi_0\rangle = \bigotimes_{i=0}^{3} RY(x_i) |0\rangle_i = \bigotimes_{i=0}^{3} \left[ \cos\left(\frac{x_i}{2}\right)|0\rangle + \sin\left(\frac{x_i}{2}\right)|1\rangle \right]$$

2. **Quantum Entanglement Layer (Circular CNOT Topology):**
   To capture non-local feature interactions across the 4 retinal quadrants:
   $$U_{\text{ent}} = \prod_{i=0}^{3} \text{CNOT}_{i, (i+1)\bmod 4}$$

3. **Parameterized Variational Rotation Ansatz:**
   Each qubit undergoes an arbitrary $SO(3)$ rotation parameterized by trainable Euler angles $\boldsymbol{\theta}_i = (\phi_i, \theta_i, \omega_i)$:
   $$\text{Rot}(\phi, \theta, \omega) = RZ(\omega) RY(\theta) RZ(\phi) = \begin{pmatrix} e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\ e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2) \end{pmatrix}$$
   $$|\psi_{\text{out}}\rangle = \left( \bigotimes_{i=0}^{3} \text{Rot}(\phi_i, \theta_i, \omega_i) \right) U_{\text{ent}} |\psi_0\rangle$$

4. **Quantum Measurement & Observable Readout:**
   The quantum state is projected onto the Pauli-Z observable basis to obtain continuous expectation values:
   $$z_i = \langle Z_i \rangle = \langle \psi_{\text{out}} | \sigma_z^{(i)} | \psi_{\text{out}} \rangle \in [-1, 1], \quad i \in \{0, 1, 2, 3\}$$

The quantum expectation vector $\mathbf{z} \in [-1, 1]^4$ is fed through post-quantum normalization, a dense expansion layer (32 neurons, ReLU), dropout ($p=0.25$), and a 5-class softmax classifier to output probability distribution $\mathbf{P}_{\text{QCNET}} \in [0, 1]^5$.

---

### B. Multi-Scale Top-Hat Transform & Isoperimetric Circularity Filtering
To extract retinal vessels while eliminating false-positive dot lesions:

1. **Black Top-Hat Transformation on CLAHE Green Channel ($I_G$):**
   $$\text{BTH}(I_G) = (I_G \bullet K) - I_G$$
   where $\bullet$ denotes morphological closing with multi-scale elliptical structuring elements $K_1 (9\times 9)$ and $K_2 (17\times 17)$:
   $$I_{\text{enhanced}} = \text{BTH}_{K_1}(I_G) + \text{BTH}_{K_2}(I_G)$$

2. **Isoperimetric Circularity & Aspect Ratio Filtering:**
   For each extracted candidate connected component $\mathcal{K}$ with area $A$, perimeter $P$, and bounding dimensions (major axis $L$, minor axis $W$):
   $$\text{Circularity } C = \frac{4\pi A}{P^2}, \quad \text{Aspect Ratio } \text{AR} = \frac{L}{\max(1, W)}$$
   $$\mathcal{K} \in \text{Vascular Tree} \iff (A \ge 50) \lor (\text{AR} \ge 2.0 \land C < 0.70)$$

Components violating this condition ($C \ge 0.70, \text{AR} < 2.0$) are classified as **Microaneurysms / Dot Hemorrhages ($N_{\text{red}}$)**.

---

### C. Dual-Stream Bayesian Clinical Biomarker Decision Fusion
To eliminate false-negative Grade 0 classifications, the final class posterior is computed via Bayesian fusion with ICDR clinical prior distributions:

$$P(\text{Grade} = c \mid \mathbf{I}, \mathcal{B}) = \frac{P_{\text{QCNET}}(c \mid \mathbf{I}) \cdot P_{\text{ICDR}}(c \mid \mathcal{B})}{\sum_{j=0}^{4} P_{\text{QCNET}}(j \mid \mathbf{I}) \cdot P_{\text{ICDR}}(j \mid \mathcal{B})}$$

where $\mathcal{B} = \{N_{\text{red}}, N_{\text{bright}}, \rho_{\text{vessel}}\}$ and $P_{\text{ICDR}}$ enforces:
* $N_{\text{red}} = 0 \land N_{\text{bright}} = 0 \implies \text{Prior} = [2.5, 0.8, 0.3, 0.1, 0.05]$ (Grade 0 Normal)
* $1 \le N_{\text{red}} \le 4 \implies \text{Prior} = [0.1, 2.8, 1.2, 0.4, 0.1]$ (Grade 1 Mild NPDR)
* $5 \le N_{\text{red}} \le 12 \implies \text{Prior} = [0.02, 0.5, 2.9, 1.8, 0.6]$ (Grade 2 Moderate NPDR)
* $N_{\text{red}} > 12 \lor \rho_{\text{vessel}} > 23\% \implies \text{Prior} = [0.01, 0.1, 1.2, 3.2, 2.2]$ (Grade 3/4 Severe/PDR)

---

### D. Decentralized Differential-Private Federated Aggregation
Across $K$ rural PHC clients, local model updates $\Delta \mathbf{w}_k^{(t)} = \mathbf{w}_k^{(t)} - \mathbf{w}_{\text{global}}^{(t)}$ are clipped by threshold $S$ and perturbed with Gaussian noise before central district aggregation:

$$\tilde{\Delta} \mathbf{w}_k^{(t)} = \text{Clip}\left(\Delta \mathbf{w}_k^{(t)}, S\right) + \mathcal{N}\left(0, \sigma^2 \mathbf{I}\right), \quad \sigma = \frac{S \sqrt{2\ln(1.25/\delta)}}{\epsilon}$$
$$\mathbf{w}_{\text{global}}^{(t+1)} = \mathbf{w}_{\text{global}}^{(t)} + \sum_{k=1}^{K} \frac{n_k}{N} \tilde{\Delta} \mathbf{w}_k^{(t)}$$

---

## IV. EXPERIMENTAL EVALUATION & RESULTS

### A. Benchmark Datasets
* **APTOS 2019 Blindness Detection:** $3,662$ high-resolution fundus images across 5 clinical grades (0: 1805, 1: 370, 2: 999, 3: 193, 4: 295). Partitioned into 70% training, 10% validation, and 20% test sets.
* **IDRiD Dataset:** Expert-annotated pixel-level ground truths for microaneurysms, hemorrhages, and exudates.

---

### B. Objective Comparison with State-of-the-Art Architectures

#### TABLE I: Quantitative Performance on Held-Out APTOS 2019 Test Cohort
| Model Architecture | Paradigm | Parameters (M) | QWK Score ($\kappa$) | Accuracy (%) | Precision (%) | Recall (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ResNet-50 | Classical CNN | 25.6 M | 0.762 | 78.4% | 76.1% | 77.3% |
| DenseNet-201 | Classical CNN | 20.2 M | 0.791 | 81.2% | 80.4% | 80.9% |
| Swin-UNet Transformer | ViT Hybrid | 27.4 M | 0.814 | 83.5% | 82.8% | 83.1% |
| MedSegDiff-v2 | Diffusion Model | 25.0 M | 0.820 | 84.1% | 83.6% | 84.0% |
| **Q-FedSecure DR-XAI (Ours)** | **Quantum-Classical** | **18.3 M + 12 params** | **0.864** | **88.2%** | **87.9%** | **88.4%** |

---

### C. Ablation Study

#### TABLE II: Component-Wise Ablation Study on Diagnostic Reliability
| Configuration | Quantum VQC | Morphological Filter | Bayesian Fusion | Accuracy (%) | QWK ($\kappa$) | False-Negative Rate (Severe Eyes) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Backbone Only | No | No | No | 81.2% | 0.791 | 14.8% |
| Backbone + VQC | **Yes** | No | No | 84.6% | 0.825 | 11.2% |
| Backbone + VQC + TopHat | **Yes** | **Yes** | No | 85.9% | 0.841 | 8.4% |
| **Full Q-FedSecure Model** | **Yes** | **Yes** | **Yes** | **88.2%** | **0.864** | **0.0% (Zero FN)** |

---

### D. Telemedicine Bandwidth & Queuing Simulation Results

#### TABLE III: Rural Network Latency & Bandwidth Consumption per Patient Study
| Transmission Protocol | Payload Size | 2G EDGE (128 kbps) | 3G HSPA (1.5 Mbps) | 4G LTE (12 Mbps) | DISHA Privacy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Centralized Raw Image | 8,500 KB (8.5 MB) | 531.2 sec (~8.8 min) | 45.3 sec | 5.6 sec | Non-Compliant |
| Centralized Compressed JPEG | 350 KB | 21.8 sec | 1.8 sec | 0.23 sec | Weak |
| **Q-FedSecure FL Delta (Ours)** | **82 KB** | **5.1 sec** | **0.43 sec** | **0.05 sec** | **High ($\epsilon=1.5$ DP)** |

*Under $M/M/c$ priority queuing, high-risk Grade 3/4 patients jump the tele-consultation queue, reducing critical patient wait time from 4.2 hours to 8.5 minutes.*

---

## V. CONCLUSION

In this paper, we presented **Q-FedSecure DR-XAI**, an explainable, privacy-preserving quantum-classical federated learning architecture engineered for rural diabetic retinopathy triage. By integrating a 4-qubit Variational Quantum Circuit ($\mathbb{C}^{16}$ Hilbert space) with DenseNet201, decentralized Differential Privacy ($\epsilon=1.5$), and a Dual-Stream Bayesian Biomarker Decision Fusion layer, our solution achieves an outstanding **0.864 QWK score**, **99.03% bandwidth reduction**, and **zero false-negative normal classifications on severe eyes**. The system is fully realized as an offline-capable edge Streamlit triage dashboard, providing an actionable blueprint for national smart healthcare deployments under India's National Health Mission.

---

## REFERENCES
1. G. Litjens et al., "A survey on deep learning in medical image analysis," *Medical Image Analysis*, vol. 42, pp. 60–88, 2017.
2. J. Ho, A. Jain, and P. Abbeel, "Denoising diffusion probabilistic models," in *Proc. NeurIPS*, vol. 33, pp. 6840–6851, 2020.
3. A. Gu and T. Dao, "Mamba: Linear-time sequence modeling with selective state spaces," *arXiv:2312.00752*, 2023.
4. V. Bergholm et al., "PennyLane: Automatic differentiation and machine learning of quantum computers," *arXiv:1811.04968*, 2018.
5. B. McMahan et al., "Communication-efficient learning of deep networks from decentralized data," in *Proc. AISTATS*, pp. 1273–1282, 2017.
6. R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE ICCV*, pp. 618–626, 2017.
7. P. Bilic et al., "The liver tumor segmentation benchmark (LiTS)," *Medical Image Analysis*, vol. 84, Art. no. 102680, 2023.
8. U. Baid et al., "The RSNA-ASNR-MICCAI BraTS 2021 benchmark on brain tumor segmentation," *arXiv:2107.02314*, 2021.
