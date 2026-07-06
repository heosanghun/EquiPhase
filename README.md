# EquiPhase: 조건부 상분리 단백질을 위한 열역학 정합형 심플렉틱 DEQ 및 검증 프레임워크

**EquiPhase** is a rigorous, pre-registered benchmarking framework designed to evaluate machine learning paradigms for condition-dependent liquid-liquid phase separation (LLPS) of intrinsically disordered proteins (IDPs) under environmental extrapolation and sequence family OOD generalization.

이 프로젝트는 본래 본질적 무질서 단백질(IDPs)의 환경 의존적 상분리 현상을 DEQ 모델로 벤치마킹하기 위해 설계되었습니다. 본 개정판에서는 기존 표준 수축성 DEQ 모델의 수축 붕괴(Contractivity Collapse)를 극복하기 위해 새롭게 구현된 **열역학 정합형 심플렉틱 DEQ(Symplectic DEQ, S-DEQ)** 및 **UPAF 누수 감사 프레임워크**에 대한 물리적 구현체와 실시간 수치 검증 내용을 기존 사전 등록 벤치마크 결과물들과 통합하여 제공합니다.

---

## ✨ 신규 업데이트: 열역학 정합형 심플렉틱 DEQ (S-DEQ) 및 UPAF 구현

### 1. 표준 DEQ의 한계 극복: 수축 붕괴 (Contractivity Collapse) 해결
기존 표준 DEQ 모델은 역전파(Backpropagation) 시 IFT(Implicit Function Theorem) 그래디언트의 안정성을 위해 Lipschitz 상수 $L < 1.0$이라는 전역 수축 조건(Contractivity)을 엄격히 만족해야 합니다. 그러나 실제 벤치마크 진단 결과, 수축 인자가 $L_{max} \approx 6.4 \gg 1.0$로 측정되는 **수축 붕괴(Contractivity Collapse)**를 보여 모든 다중 시작점이 하나의 고정점으로 붕괴하고 노이즈 낀 그래디언트가 흐르는 현상이 확인되었습니다.

이를 해결하기 위해, 위상 공간(Phase Space)의 부피를 수학적으로 보존하는 **해밀토니안 심플렉틱 동역학(Symplectic Dynamics)**과 리프프로그 적합기(Leapfrog Integrator)를 물리 엔진으로 도입한 `SymplecticDEQ`를 신규 구현하였습니다. 이를 통해 인공지능이 강제 수축 없이 다중 안정성 에너지 곡면(Bistable/Multistable Landscape)을 온전히 보존하며 평형 상태를 탐색할 수 있습니다.

### 2. 동역학 수치 계산의 수학적 검증 완료 (`verify_mathematics.py`)
* **부피 보존성 (리우빌 정리):** 마찰이 없을 때 야코비안 행렬식 $\det(J) = 1.0000000$ (오차 범위 $< 10^{-7}$)으로 정확히 위상 공간 부피를 보존합니다.
* **물리적 마찰 감쇄:** 마찰 인자(damping = 0.2)를 주었을 때 행렬식이 이론적 기대값인 $(1 - \gamma)^{half\_dim} = 0.4096000$에 정확하게 일치하여 소실됨을 증명하였습니다.
* **Krylov 고유값 분산 계산:** 켤레복소수 쌍의 고유값 반지름을 2단계 Krylov 투영법으로 정확하게 역산(`0.7997` vs 실제 `0.8000`)해 냅니다.
* **이중 우물 포텐셜 추적:** 1차원 이중 우물 물리 다리 테스트를 거쳐 공존하는 2개의 안정된 근($z^* = \pm 1.0$, 마진 $0.10$)과 1개의 불안정한 에너지 장벽($z^* = 0.0$, 마진 $-0.05$)을 동적으로 포착하였습니다.

### 3. UPAF를 통한 지름길 학습(Shortcut Learning) 감시
* 모델이 단백질의 진짜 열역학적 에너지 상태를 배우지 않고 서열 길이나 기하학적 RMSD 등 단순 편법에 의존하는 것을 방지하기 위해 플레이스보 축(Placebo Audit Axis)을 통한 누수 경보 장치(UPAF)를 설계하고 WBC 바이오 데이터셋 상에서 이를 완벽히 실증하였습니다.

---

## 📌 Key Scientific Findings & Verdicts (핵심 과학적 발견 및 평정)

The project evaluates two primary tracks:
1. **Track-1 (Hypothesis H1):** Testing whether implicit fixed-point representations via Deep Equilibrium (DEQ) models improve condition-dependent LLPS classification under out-of-distribution (OOD) salt conditions. (**Verdict: Rejected / NULL**)
2. **Phase 5 (Hypothesis H2):** Evaluating if per-residue PLM embeddings, attention pooling, explicit biophysical descriptors, and monotonic concentration constraints can enable sequence family generalization under low-salt conditions. (**Verdict: Rejected / NULL**)

### 1. Hypothesis H1: Implicit Inductive Bias for Salt Extrapolation (Phase 3)
* **Hypothesis:** Implicit fixed-point layers (DEQs) enhance generalization under out-of-distribution (OOD) salt conditions (Train: $\le 150$ mM, Test: $> 300$ mM).
* **Verdict:** **Formal Rejection (NULL).** 
* **Results:** The DEQ candidate underperformed all baselines (Validation AUPRC 0.6570, Locked Test AUPRC 0.6078), performing worse than a simple condition-aware MLP and finite-depth recurrent unrolled models ($K=8$ unroll achieved Val AUPRC 0.7701, Test AUPRC 0.6503).
* **Diagnosis:** **Contractivity Collapse.** Despite spectral normalization and residual damping, the residual cell Jacobian spectral norm remained $L_{max} \approx 6.4 \gg 1.0$. This violated the contractivity requirement of the Implicit Function Theorem (IFT), yielding corrupted and noisy backpropagation gradients.

### 2. Hypothesis H2: Feature-Driven Sequence Family Generalization (Phase 5)
* **Hypothesis:** Biophysical descriptors + attention pooling over ESM-2 residue-level embeddings, combined with monotone solute constraints, enable generalization above the no-skill baseline on unseen sequence families.
* **Verdict:** **Formal Rejection (NULL).** 
* **Results:** 
  * **AttentionMLP (Pre-registered Winner):** Cleared the validation baseline (AUPRC **0.7883** vs. **0.6812** no-skill) but failed to confirm on the locked test set (AUPRC **0.7697**, 95% CI: `[0.6264, 0.8825]`, lower bound 0.6264 < 0.6448 no-skill) due to statistical power constraints (only 26–27 families per split).
  * **Tab-Monotone XGBoost (Exploratory / Off-Protocol):** Failed validation clearance (CI lower bound 0.6660 < 0.6812) and was ineligible for confirmatory testing. On the spent locked test set, it achieved an exploratory AUPRC of **0.8313** (95% CI: `[0.7024, 0.9190]`, AUROC **0.7234**).
* **Takeaways:**
  * **Label-Noise Bottleneck:** We diagnosed that **45.71%** of low-salt database records contain conflicting labels for identical sequences due to starting solute concentration variation, capping classifier performance.
  * **Monotonic Constraints as a Future Lead:** Enforcing positive monotonic constraints on concentration is a promising lead to bypass label noise, but the current XGBoost success is post-hoc and exploratory, requiring a new pre-registration and fresh test set to validate.

---

## 📊 Comprehensive Experimental Results (종합 실험 결과)

### Phase 3 Benchmark (Salt Extrapolation - H1)
Evaluated across **5 seeds** (`42, 100, 2026, 777, 999`) using cached sequence embeddings from the standard `esm2_t33_650M_UR50D` backbone. Confidence intervals (95% CI) computed using **cluster block bootstrapping** (1000 iterations).

| Model Architecture | Val AUPRC | Val AUROC | Test AUPRC (Locked) | Test AUROC (Locked) | Peak VRAM | Wall-Clock (per seed) | Final Lipschitz $L_{max}$ | Final Solver Residual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ablation A: K=8** | **0.7701** | 0.6154 | **0.6503** | 0.5646 | 64.69 MB | 24.84s | 9.3981 | 0.6230 |
| **Ablation A: K=matched (K=40)** | 0.7675 | 0.6151 | 0.6523 | 0.5695 | 64.89 MB | 104.25s | 4.6762 | 0.2087 |
| **Ablation A: K=16** | 0.7620 | 0.6145 | 0.6461 | 0.5668 | 64.94 MB | 44.29s | 6.4138 | 0.4081 |
| **Condition-aware MLP** | 0.7295 | 0.5847 | 0.6396 | 0.5562 | 21.80 MB | 3.01s | - | - |
| **ESM-2 embedding + MLP** | 0.7248 | 0.5759 | 0.6385 | 0.5469 | 21.48 MB | 2.99s | - | - |
| **DiG-inspired EBM** | 0.7161 | 0.5703 | 0.6463 | 0.5632 | 21.80 MB | 3.00s | - | - |
| **DEQ (candidate)** | 0.6570 | 0.4714 | 0.6078 | 0.5124 | 62.86 MB | 213.81s | 6.4267 | 0.0144 |
| **Ablation B (no cond. coupling)** | 0.6463 | 0.4689 | 0.6195 | 0.4915 | 62.04 MB | 213.10s | 7.4918 | 0.0716 |

* **Validation No-Skill Baseline AUPRC:** 0.7230
* **Locked Test No-Skill Baseline AUPRC:** 0.6840

---

### Phase 5 Benchmark (Family Extrapolation - H2)
Evaluated under low-salt conditions ($\le 150$ mM) on family-disjoint splits constructed using Jaccard 3-mer proxy clustering.

* **Validation Set (No-Skill Baseline AUPRC: 0.6812)**
  * **AttentionMLP (Pre-registered Winner):** AUPRC **0.7883** (95% CI: `[0.6885, 0.8822]`) | **Clears Baseline: YES**
  * **Tab-Monotone XGBoost (Exploratory):** AUPRC **0.7742** (95% CI: `[0.6660, 0.8895]`) | **Clears Baseline: NO**
* **Locked Test Set (No-Skill Baseline AUPRC: 0.6448)**
  * **AttentionMLP (Confirmatory):** AUPRC **0.7697** (95% CI: `[0.6264, 0.8825]`) | **Clears Baseline: NO**
  * **Tab-Monotone XGBoost (Exploratory / Off-Protocol):** AUPRC **0.8313** (95% CI: `[0.7024, 0.9190]`) | **Clears Baseline: YES** (AUROC: **0.7234**, 95% CI: `[0.6251, 0.8271]`)

---

## 🛠️ Codebase Structure (코드베이스 구조)

```
c:/Project/EquiPhase/ (Virtual mapped as D:/AI/EquiPhase/)
├── equiphase/
│   ├── models/
│   │   ├── symplectic_deq.py         # [NEW] 해밀토니안 리프프로그 통합 Symplectic DEQ 모델
│   │   └── spectral_dispatch.py      # [NEW] 켤레복소수 포착용 2단계 Krylov 고유값 분산 계산기
│   ├── baselines/
│   │   ├── train_baselines.py        # Baseline models training (Phase 2 & 3)
│   │   ├── train_phase5.py           # Pre-registered validation modeling for Phase 5
│   │   └── test_phase5_evaluation.py # Locked test evaluation script for Phase 5
│   ├── data/
│   │   ├── biophysical.py            # Biophysical feature calculator (FCR, NCPR, SCD, etc.)
│   │   ├── make_new_splits.py        # Sequence clustering & split creation
│   │   ├── precompute_residue_esm2.py# Length-sorted dynamic ESM-2 batch precomputation
│   │   └── raw/                      # Raw LLPSDB v2.0 xls files
│   └── eval/
│       └── metrics.py                # Pairwise CI and cluster block bootstrap metrics
├── PRE_REGISTRATION.json             # Locked Phase 3 configuration
├── PRE_REGISTRATION_PHASE5.json      # Locked Phase 5 configuration
├── DEVIATIONS.md                     # Immutable pre-registration deviations log (R1)
├── final_report.md                   # Canonical final project report
├── task.md                           # Task checklist tracking file
├── walkthrough.md                    # In-depth technical walkthrough of results
├── verify_mathematics.py             # 심플렉틱 동역학 및 수치 부피보존 독립 검증 스크립트
├── test_iss_module.py                # ISS 수학적/그래디언트 흐름 종합 검증 테스트 스위트
└── upaf.py                           # 플레이스보 축 연동 데이터 누수 방지 모듈
```

---

## ⚙️ How to Run & Replicate (실행 및 재현 방법)

### 1. Prerequisites (준비 사항)
Ensure you have Python 3.10+ and a CPU or CUDA-capable GPU. Install the required dependencies:
```bash
# CPU전용 PyTorch 및 torchdeq 설치 (수치 해석 속도 및 재현성을 극대화)
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install torchdeq xgboost scikit-learn pandas numpy openpyxl matplotlib pillow
```

### 2. Run Physical & Mathematical Verification (신규)
심플렉틱 동역학 모델의 에너지 보존(Liouville 정리) 및 고유값 반지름 역산 동작을 검증합니다:
```bash
.venv\Scripts\python.exe verify_mathematics.py
```

### 3. Run ISS Neural Network and IFT Gradient Test (신규)
암시적 고정점 탐색 수렴성 및 암시적 함수 정리(IFT) 기반 오차 역전파 그래디언트 분석 테스트를 수행합니다:
```bash
.venv\Scripts\python.exe test_iss_module.py
```

### 4. Precompute Features (기존 기능)
To re-generate the biophysical features and ESM-2 embeddings:
```bash
# Compute Sequence Biophysical Descriptors
.venv\Scripts\python.exe equiphase/data/precompute_biophysical.py

# Precompute ESM-2 Residue-Level Embeddings
.venv\Scripts\python.exe equiphase/data/precompute_residue_esm2.py
```

### 5. Run Phase 5 Training & Validation
To train the models on `train_phase5.tsv` and evaluate on `val_phase5.tsv`:
```bash
.venv\Scripts\python.exe equiphase/baselines/train_phase5.py
```

### 6. Run Locked Test Set Evaluation
To train the final models (across 5 seeds) and generate locked test results on `test_phase5.tsv`:
```bash
.venv\Scripts\python.exe equiphase/baselines/test_phase5_evaluation.py
```
The output will be saved directly to `locked_test_results_phase5.json`.

---

## 🛡️ Scientific Rigor & Rules Audited (과학적 연구 규정 및 무결성)
We adhere to strict methodological rules to prevent data contamination and p-hacking:
1. **Rule R1 (Immutable Pre-registration):** Configurations and hypothesis targets were frozen in JSON files. All modifications (such as updating the target to binary classification due to starting-concentration ambiguity) are documented as dated entries in `DEVIATIONS.md`.
2. **Rule R2 (No Unverified Numbers):** All reported performance metrics are computed using 1,000-iteration cluster block bootstrap resampling over sequence families to preserve correlation structures.
3. **Rule R3 (Anti-Recycling Registry):** The evaluation harness hashes all training losses and metrics. Duplicate metrics from hardcoded values or copy-paste errors trigger a `RecyclingFabricationException` to prevent fraud.
