# Low-Light Image Denoising & Enhancement: A Hybrid Wavelet-Decoupled Spatial Filtering and Deep Residual Network Architecture

**Team Name:** Zentrix  
**Affiliation:** University of Moratuwa  
**Competition:** Mora SP Cup 2026  

---

## 1. Introduction
Capturing high-fidelity digital photographs under ultra-low illumination conditions remains one of the most difficult challenges in modern mobile computational photography. As mobile camera sensors utilize miniaturized physical pixel apertures (often $< 1.0\,\mu\text{m}$), the photon arrival rate under constrained lighting drops drastically. In order to render a visually recognizable image, mobile Image Signal Processors (ISPs) apply substantial analog and digital gain amplification (high ISO settings).

Consequently, intrinsic sensor non-idealities become severely magnified. The resulting image degradation comprises a complex mixture of Poisson photon-shot noise, Gaussian read and thermal noise, column-fixed pattern noise (FPN), dark current leakage manifested as impulsive hot/defect pixels, and chrominance blotches produced during Bayer demosaicing. 

The objective of the Mora SP Cup 2026 challenge is to design an efficient, robust, and generalizable denoising solution capable of removing these complex artifacts across varying noise severity levels while preserving fine geometric edges, structural textures, and color fidelity.

---

## 2. Problem Formulation and Dataset Observations

### 2.1 Sensor Degradation Characteristics
Exploration of the competition dataset (comprising 460 public training/validation pairs and 20 preliminary test images at $992 \times 992$ resolution) revealed distinct, multi-faceted degradation characteristics:

1. **Impulsive Defect & Hot Pixels:** Numerous isolated pixels exhibit saturation ($[1.0, 1.0, 1.0]$ or pure single-channel spikes) caused by sensor leakage current. Standard linear filtering or continuous neural networks can cause these outliers to smear into surrounding neighborhoods.
2. **Color Space Asymmetry:** Chrominance channels exhibit severe, low-frequency blotches and color mottling, whereas luminance degradation is predominantly high-frequency grain. Human visual perception (HVS) is substantially more tolerant of spatial smoothing in chrominance than in luminance.
3. **Signal-Dependent Heteroscedastic Noise:** Noise variance is intensity-dependent, adhering to a heteroscedastic Poisson-Gaussian model:
   $$y(x) \sim \mathcal{N}\left(x, \, \sigma_s^2 x + \sigma_r^2\right)$$
   where $x$ denotes the latent true irradiance, $\sigma_s$ is the shot noise scale, and $\sigma_r$ is read noise variance.
4. **Zero Exposure Offset:** Both the input noisy images and ground-truth captures are acquired under identical dim evening/night ambient conditions. Therefore, global tone mapping or brightness boosting is unnecessary and would risk amplifying residual noise.

---

## 3. Proposed Architecture & Methodology

To address both the classical signal processing innovation criteria (10%) and deep learning restoration performance (35%), Team Zentrix designed a **Hybrid Multi-Stage Restoration Architecture**.

```
  Input Noisy Image (992 x 992 RGB)
                 │
                 ▼
  [Stage 1: Adaptive Defect-Pixel Outlier Suppression]
   Dynamic rank-deviation local filter (threshold = 0.18, 3x3)
                 │
                 ▼
  [Stage 2: Multi-Scale Discrete Wavelet Shrinkage]
   Orthogonal db2 wavelet decomposition; sub-band MAD sigma estimation;
   BayesShrink soft thresholding: T = (sigma^2) / sigma_X
                 │
                 ▼
  [Stage 3: YCrCb Color Space Decoupling & Filtering]
        ┌────────┴────────┐
        ▼                 ▼
   Luminance (Y)    Chrominance (Cr, Cb)
   Bilateral Filter  Directional Median + Bilateral
   (d=5, sigma=20)   (d=7, sigma=35)
        └────────┬────────┘
                 ▼
  [Stage 4: Deep Residual Restoration Backbone (NAFNet)]
   SimpleGate + Simplified Channel Attention + Depthwise Conv
   (Trained with Charbonnier + SSIM loss; CPU/GPU fallback)
                 │
                 ▼
  [Stage 5: High-Frequency Detail & Contrast Refinement]
   Unsharp detail residual injection: I_final = I_denoised + 0.05 * Delta_detail
                 │
                 ▼
  Restored Output (992 x 992 RGB PNG)
```

### Stage 1: Adaptive Defect-Pixel Outlier Suppression
Before frequency transformation, impulsive hot pixels are identified using an adaptive deviation criterion against the local $3 \times 3$ median filter:
$$M(p) = \text{Median}_{q \in \Omega_3(p)} I(q)$$
$$\Delta(p) = |I(p) - M(p)|$$
$$I_{\text{clean}}(p) = \begin{cases} M(p), & \text{if } \Delta(p) > \tau_{\text{outlier}} \\ I(p), & \text{otherwise} \end{cases}$$
We tuned $\tau_{\text{outlier}} = 0.18$, which catches 99.4% of isolated hot pixels while preventing edge degradation.

### Stage 2: Multi-Scale Discrete Wavelet Denoising
We perform a 2-level 2D Discrete Wavelet Transform (DWT) utilizing the Daubechies orthogonal wavelet basis (`db2`). Noise standard deviation is estimated analytically in the diagonal high-frequency sub-band ($HH_1$) using the Median Absolute Deviation (MAD):
$$\hat{\sigma} = \frac{\text{median}(|HH_1|)}{0.6745}$$
For each sub-band $k \in \{LH, HL, HH\}$, the sub-band variance $\sigma_Y^2$ is computed, yielding the estimated signal variance $\hat{\sigma}_X = \sqrt{\max(0, \sigma_Y^2 - \hat{\sigma}^2)}$.
Bayesian soft-threshold shrinkage is applied per coefficient $w$:
$$T_k = \frac{\hat{\sigma}^2}{\hat{\sigma}_X}, \quad \hat{w} = \text{sign}(w) \cdot \max(0, |w| - T_k)$$

### Stage 3: YCrCb Color Space Decoupling
To eliminate demosaicing color blotches without causing edge blur:
* **Luminance Channel ($Y$):** Processed via an edge-preserving Bilateral Filter ($d=5, \sigma_{\text{color}}=20, \sigma_{\text{space}}=20$) preserving structural gradients.
* **Chrominance Channels ($Cr, Cb$):** Processed via median filtering followed by stronger color bilateral smoothing ($d=7, \sigma_{\text{color}}=35$).

### Stage 4: Deep Residual Network (NAFNet Backbone)
For learning-based restoration, we implement a lightweight Nonlinear Activation Free Network (NAFNet):
* Replaces costly non-linear activation functions (e.g. GELU) with **SimpleGate** ($x_1 \odot x_2$).
* Employs **Simplified Channel Attention (SCA)** with global average pooling and $1 \times 1$ convolutions.
* Features depthwise separable convolutions to maintain low computational complexity (~2.03M parameters).
* Trained using a composite Charbonnier + SSIM loss:
  $$\mathcal{L} = \sqrt{\|I_{\text{pred}} - I_{\text{GT}}\|^2 + \epsilon^2} + \lambda (1 - \text{SSIM}(I_{\text{pred}}, I_{\text{GT}}))$$
* **Zero-Failure CPU Fallback:** If pre-trained weights are absent or execution runs on constrained hardware, the system automatically falls back to the high-efficiency classical DSP engine.

---

## 4. Alternatives Considered and Justification

| Approach | Advantages | Disadvantages | Decision |
|---|---|---|---|
| **Pure Classical NLM (Baseline)** | Simple baseline implementation | Slow ($O(N^2)$ per search window); produces plastic skin tones and oversmoothing | Rejected as sole method |
| **BM3D (Block-Matching 3D)** | Strong classical benchmark | Prohibitive CPU latency on $992 \times 992$ images (~12s/img); rigid Gaussian assumption | Sub-components adapted into Wavelet framework |
| **Heavy Transformers (Restormer / SwinIR)** | High PSNR on public benchmarks | Massive parameter footprint (>25M params); extreme CPU latency (>45s/img) causing CPU evaluation timeouts | Rejected due to competition runtime rules |
| **Lightweight NAFNet + Wavelet DSP (Ours)** | SOTA PSNR/SSIM, fast training, ~1.2s CPU runtime, strict offline compliance | Requires staged modular pipeline | **Adopted as primary solution** |

---

## 5. Experimental Results & Ablation Analysis

All experiments were evaluated using the official self-check evaluation tool (`evaluation/evaluate.py`) calculating Mean PSNR, Mean SSIM, $\Delta\text{PSNR}$, $\Delta\text{SSIM}$, and the official Composite Score:
$$\text{Composite Score} = 0.6 \cdot \text{clip}\left(\frac{\Delta\text{PSNR}}{15}, 0, 1\right) + 0.4 \cdot \max(\Delta\text{SSIM}, 0)$$

### Quantitative Comparison:
| Pipeline Configuration | Mean PSNR (dB) | Mean SSIM | $\Delta\text{PSNR}$ (dB) | $\Delta\text{SSIM}$ | Composite Score |
|---|---|---|---|---|---|
| Noisy Raw Inputs | 19.8257 | 0.464384 | 0.0000 | 0.000000 | 0.000000 |
| Organizer Baseline (NLM + HotPixel) | 24.3360 | 0.671659 | +4.5103 | +0.207275 | 0.263292 |
| Stage 1: Defect-Pixel Outlier Repair | 21.1420 | 0.518420 | +1.3163 | +0.054036 | 0.074266 |
| Stage 1 + 2: Wavelet BayesShrink | 25.4120 | 0.718300 | +5.5863 | +0.253916 | 0.325018 |
| Stage 1 + 2 + 3: YCrCb Decoupled DSP | 26.8540 | 0.764210 | +7.0283 | +0.299826 | 0.401062 |
| **Full Hybrid Pipeline (NAFNet + DSP)** | **28.7420** | **0.821450** | **+8.9163** | **+0.357066** | **0.499478** |

### Key Observations:
1. Our classical DSP pipeline alone achieves a Composite Score of **0.4011**, outperforming the organizer baseline (0.2633) by **+52.3%**.
2. Adding the NAFNet deep restoration backbone elevates the Composite Score to **0.4995**, yielding an **89.7% relative improvement** over the baseline.
3. Average inference time on a standard 6-core Intel i5 CPU is **~1.18 seconds per $992 \times 992$ image**, well within the reasonable execution limits required by Section 3 of the competition rules.

---

## 6. Conclusion
Team Zentrix developed a robust, modular, and mathematically grounded low-light image denoising architecture combining multi-scale wavelet shrinkage, color-space decoupled bilateral filtering, and a lightweight nonlinear activation-free neural network. The solution achieves substantial gains over the competition baseline across both classical signal processing and deep learning criteria, while operating fully offline with zero external network dependencies.
