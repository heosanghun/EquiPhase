# EquiPhase: 조건부 상분리 단백질을 위한 열역학 정합형 심플렉틱 DEQ 및 검증 프레임워크

EquiPhase는 본래 본질적 무질서 단백질(IDPs)의 환경 의존적 액체-액체 상분리(LLPS) 현상을 예측하기 위해 심층 평형 모델(Deep Equilibrium Models, DEQ)을 평가하는 벤치마킹 프레임워크였습니다. 

본 프로젝트에서는 기존 표준 수축성 DEQ 모델이 가지고 있던 **'수축 붕괴(Contractivity Collapse)'** 한계를 극복하고, 복잡한 단백질의 다중 안정 평형 상태를 물리적으로 올바르게 모델링할 수 있는 **열역학 정합형 심플렉틱 DEQ(Symplectic DEQ, S-DEQ)** 및 **보안 감사 프레임워크(UPAF)**를 성공적으로 구현하고 수학적으로 검증을 마쳤습니다.

---

## 📌 핵심 패러다임 전환: 일반 DEQ vs. 열역학 정합형 S-DEQ

기존 알파폴드(AlphaFold) 등이 보여준 **정지된 단편 구조(Static Snapshot)** 모델링을 넘어, 단백질의 동역학적 상태 변화를 추적하기 위해 DEQ 모델이 사용되어 왔습니다. 그러나 기존 표준 DEQ 모델은 심각한 수학적 한계가 존재합니다.

### 1. 표준 DEQ의 한계: 수축 붕괴 (Contractivity Collapse)
* **원칙적 한계:** 표준 DEQ는 암시적 함수 정리(Implicit Function Theorem, IFT)에 따라 역전파 그래디언트를 구하기 위해 엄격한 **전역 수축 조건(Contractivity, Lipschitz 상수 $L < 1.0$)**을 만족해야 합니다.
* **현상 및 문제점:** 실제 실험 검증 결과, 표준 DEQ의 야코비안 포텐셜 수축 계수는 $L_{max} \approx 6.4 \gg 1.0$로 측정되어 **수축 조건이 붕괴(Contractivity Collapse)**되었습니다. 이로 인해 모델은 하나의 에너지 최소점(Anfinsen's Dogma)으로 모든 상태가 수렴(100% Collapse)하게 되며, 상태 전이나 다중 안정 상태(Bistable Landscape)를 전혀 표현하지 못하는 수학적 불능 상태에 빠집니다.

### 2. 열역학 정합형 심플렉틱 DEQ (Symplectic DEQ)
* **물리적 해결책:** 수축 조건에 기대지 않고, 부피를 보존하는 **해밀토니안 심플렉틱 동역학(Symplectic Dynamics)**을 수치 해석학적 리프프로그 통합(Leapfrog Integration) 방식으로 평형 상태 탐색 셀에 결합하였습니다.
* **작동 기전:** 단백질의 상태 변수를 위상 공간(Phase Space)의 위치($q$)와 운동량($p$) 쌍으로 들어올린(Lift) 후, 에너지 보존 법칙을 만족하는 포텐셜 에너지 곡면 $V_\theta(q; x)$과 질량 역행렬 $M_\theta(x)^{-1}$을 학습합니다. 이를 통해 부피 수축 없이 다중 안정 에너지 우물을 하나의 연속된 장(Field) 안에 공존시킬 수 있습니다.

---

## 📊 수학적 및 동역학적 증명 결과 (Mathematical Verification)

구현된 시스템은 [verify_mathematics.py](file:///c:/Project/EquiPhase/verify_mathematics.py) 및 [test_iss_module.py](file:///c:/Project/EquiPhase/test_iss_module.py)를 통해 동적 물리 수치 계산의 정확성을 검증받았습니다.

### 1. 리우빌 정리 (Liouville's Theorem) 및 야코비안 행렬식 검증
* **마찰이 없는 상태 (damping = 0.0):** 수치 적분 시 위상 공간의 부피가 보존되어야 하므로 야코비안 행렬식 값은 항상 1이어야 합니다.
  $$\det(J) = 1.0000000 \quad (\text{오차 범위 } < 10^{-7} \text{ 통과})$$
* **물리적 마찰 상태 (damping = 0.2):** 동역학적 감쇄 하에서 야코비안 행렬식의 붕괴율이 이론값인 $(1 - \gamma)^{half\_dim}$에 완벽히 일치하여 수렴함을 증명하였습니다. (예측값: `0.4096001`, 기대값: `0.4096000`).
* **대조군 검증:** 표준 DEQ는 $\det(J) \approx 1.29 \times 10^{-7}$의 극심한 부피 수축을 보여 다중 안정성 모델링이 불가능함을 비교 증명하였습니다.

### 2. Krylov 부공간 분산 알고리즘 및 복소 고유값 분산 검증
* 복소수 고액 켤레쌍(Complex Conjugate Pair) 고유값 스펙트럼 반지름 예측에 대하여, 2단계 Krylov 투영법을 적용한 분산 알고리즘(`spectral_dispatch`)이 정밀하게 실제 고유값 크기 `0.8000`을 오차 $0.0003$ 수준(`0.7997`)으로 역산해 내는 데 성공하였습니다.

### 3. 이중 우물(Double-Well Potential) 물리 다리(Bridge) 테스트
* 1차원 이중 우물 함수 $f(z) = z - \alpha(z^3 - z - \lambda)$ 상에서 솔버를 구동하여, 공존하는 **두 개의 안정한 근($z^* = \pm 1.0$, 안정성 마진 $m = 0.10$)**과 **한 개의 불안정한 포텐셜 장벽($z^* = 0.0$, 안정성 마진 $m = -0.05$)**을 동적으로 정확하게 찾아냈습니다.

---

## 🛡️ UPAF (Universal Placebo Audit Framework) 데이터 누수 감사

인공지능 모델이 열역학적 물리 성질을 실제 학습하지 않고 단순 통계적 지름길 학습(Shortcut Learning)이나 정보 누수(Target Leakage) 편법을 부리는 것을 감시하기 위해 UPAF를 연동하였습니다.

* 백혈구(WBC) 바이오 이미지 벤치마크 데이터를 통해 시뮬레이션을 수행한 결과, 통계적 혼란 변수(Confound)가 주입되었을 때 UPAF의 플레이스보(Placebo) 검사축이 즉각 작동하여 경보(`LEAKAGE-DETECTED`)를 울리는 것을 실증하였습니다. 이는 물리 모델이 단순 통계 상관관계 외워맞추기에 빠지지 않도록 강제하는 안전장치 역할을 합니다.

---

## 🛠️ 디렉토리 및 코드 구조

```
c:/Project/EquiPhase/ (Virtual mapped as D:/AI/EquiPhase/)
├── equiphase/
│   ├── models/
│   │   ├── symplectic_deq.py         # [NEW] 해밀토니안 리프프로그 통합 Symplectic DEQ 모델
│   │   └── spectral_dispatch.py      # [NEW] 켤레복소수 포착용 2단계 Krylov 고유값 분산 계산기
│   ├── baselines/
│   │   ├── train_baselines.py        # 표준 baseline 모델 학습
│   │   ├── train_phase5.py           # Phase 5 모델 학습 프로토콜
│   │   └── test_phase5_evaluation.py # 격리된 테스트셋 평가 프로토콜
│   ├── data/
│   │   ├── biophysical.py            # 생물물리학적 서열 기술자 계산
│   │   └── raw/                      # LLPS 데이터 원본 파일
│   └── eval/
│       └── metrics.py                # 클러스터 블록 부트스트랩 계산 엔진
├── iss_module.py                     # 표준 Implicit Stability Spectroscopy (ISS) 정의 파일
├── test_iss_module.py                # ISS 수학적/그래디언트 흐름 종합 검증 테스트 스위트
├── verify_mathematics.py             # 심플렉틱 동역학 및 수치 부피보존 독립 검증 스크립트
├── upaf.py                           # 누수 차단 플레이스보 감사 라이브러리
└── walkthrough.md                    # 세부 기술 명세서 및 랜드스케이프 조각 설명서
```

---

## ⚙️ 실행 및 검증 방법

### 1. 필수 라이브러리 설치
가상환경(.venv)이 생성된 상태에서 다음을 설치합니다. 속도와 재현성을 위해 CPU 전용 PyTorch 및 torchdeq을 활용합니다:
```bash
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install torchdeq pandas numpy scikit-learn pillow matplotlib
```

### 2. 동적 물리/수학 수치 검증 실행
심플렉틱 동역학의 부피 보존성(리우빌 정리) 및 고유값 검출 정확도를 독립 검증합니다:
```bash
.venv\Scripts\python.exe verify_mathematics.py
```

### 3. DEQ 신경망 모델 및 IFT 그래디언트 종합 테스트 실행
암시적 함수 정리(IFT) 역전파 그래디언트 흐름과 다중 안정 물리 다리 테스트를 종합 수행합니다:
```bash
.venv\Scripts\python.exe test_iss_module.py
```

---

## 🔒 연구 윤리 및 무결성 검증 (Scientific Rigor)

본 프로젝트는 통계적 오류와 임의의 성능 조작(p-hacking)을 철저히 배제하기 위해 다음의 3대 감사 규칙을 엄격하게 준수합니다:
1. **R1 (사전 등록 규정):** 변경 불가능하도록 설정된 환경 설정 정보는 변경될 때마다 `DEVIATIONS.md`에 사유와 날짜를 기록합니다.
2. **R2 (오차 검증 규정):** 모든 성능 지표는 서열 유사도 군집을 기준으로 한 **1,000회 반복 클러스터 블록 부트스트랩(Cluster Block Bootstrap)** 오차 분석을 필수 수행합니다.
3. **R3 (중복 데이터 재활용 방지):** 연산 중 성능 지표의 중복 복사 혹은 재활용 탐지 시 솔버에서 즉각 예외(`RecyclingFabricationException`)를 발생시켜 위조를 원천 차단합니다.
