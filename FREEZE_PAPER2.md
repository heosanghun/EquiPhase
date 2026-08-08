# 🧊 FREEZE_PAPER2.md — EquiPhase DEQ (Paper 2) Status & Audit History

**Date**: 2026-08-08  
**Status**: ✅ UNFROZEN & VERIFIED (PREREGISTRATION GATES G1–G7 PASS, CONDITIONALLY ADJUDICATED)  
**Verification Seal**: Claude Independent Auditor Sealed Script (`claude_paper2_sealed_audit.py`, SHA-256 `68a2991e043984cd1ca7637441883a3fb1d82c08fc21b2b43538932028894833`)  
**Architecture Name**: EquiPhase DEQ / 32D Anisotropic Double-Well DEQ (Integrator: damped velocity Verlet)

---

## 1. 동결 해제 경과 (Unfreeze Timeline & Adjudication)

1. **사전등록 v3 봉인 (Commit `97cd2b5`)**: Preregistration Specification v3 locked prior to seed-7777 execution.
2. **실측 실행 및 체크포인트 보존 (Commit `e148866`)**: `train_paper2_deq_supervised.py` execution generated `supervised_deq_model_seed7777.pt` (SHA-256 `c6b64ec3...`) and `trajectory_basins_seed7777.csv` (SHA-256 `4cf32e...`).
3. **제3자 해시 봉인 감정 (2026-08-08)**: 외부 감사자(Claude)가 작성한 봉인 감사 스크립트 실행 결과, **전 7개 사전등록 게이트(G1–G7) 100% PASS** 입증.
   - $\rho(J_f)$ 실측 스펙트럼 $0.962738$이 감사자 정정 이론값 $\rho = 0.96273$과 소수점 4자리 완전 일치.
   - `Compare-Object` 2회 재현 diff 0 (공백) 입증.
   - 세대 간 `SEC 5` CPU 봉인 초기값 114행 비트 단위 100% 동일성 확인.

---

## 2. 논문 정본 수치표 (Official Verified Table)

| 게이트 | 실측 수치 | 사전등록 기준 | 최종 판정 |
| :--- | :--- | :--- | :---: |
| **G1** (힘 비대칭도) | ratio $\le 1.50 \times 10^{-9}\%$ | 구조적 (실측 확인: $0.0 \sim 2.63 \times 10^{-9}$) | **PASS** |
| **G2'** (상반 정합성) | $c=0.8000000\pm 5\times 10^{-8}, R\le 1.48\times 10^{-7}$ | $R < 1 \times 10^{-6}$ | **PASS** |
| **G3'** (수렴 잔차) | mean $2.565\times 10^{-10}$ / max $6.88\times 10^{-9}$ | $< 1 \times 10^{-6}$ | **PASS** |
| **G4a'** (수렴군집 수) | $N=2$ ($\rho=0.9626 / 0.9628 < 1$) | $N \ge 2$ | **PASS** |
| **G4b'** (수렴 비율) | $1.00$ (원 시드) / $0.99$ (신규 시드, 1 발산) | $\ge 0.90$ | **PASS** |
| **G5'** (최소점 변위) | $\alpha$별 $5.95, 4.87, 4.18 \times 10^{-3}$ | $\le 6.25 \times 10^{-3}$ | **PASS** |
| **G6'** (안장점 변위) | $\alpha$별 $8.89, 8.60, 8.48 \times 10^{-3}$ | $\le 1.67 \times 10^{-2}$ | **PASS** |
| **G7'** (장벽 높이) | $\Delta V = 0.230111$ (오차 $2.61 \times 10^{-3}$) | $0.2275 \pm 0.0100$ | **PASS** |
| **$\|\nabla V_{\text{net}}\|$** | $6.65\times 10^{-3}$ / $6.60\times 10^{-3}$ | $\le 1.0 \times 10^{-2}$ (가정) | **HOLDS** |

### ⚠️ 필수 동반 공개 사항 (Mandatory Disclosures & Limitations)
> [!IMPORTANT]
> 1. **제3자 자기검증 구조**: 감사 스크립트는 외부 감사 AI(Claude)가 작성·해시 봉인하였으며, 피감사 에이전트는 무수정 실행하고 자기해시 및 2회 재현, 세대 간 교차 검증으로 무결성을 담보함.
> 2. **사전등록 이탈 (Deviation)**: 역전파 사양은 사전등록상 IFT 해석적 역전파였으나, 실제 구현은 100스텝 forward solver unrolling으로 실행됨.
> 3. **수치 적분자 명칭**: 본 모델의 정확한 수치 적분자는 **damped velocity Verlet**임.
> 4. **파라미터한계**: $\alpha=1.2$에서 $\alpha$별 이론한계($4.167 \times 10^{-3}$)를 $1.0 \times 10^{-5}$ 만큼 초과하나, 사전등록 봉인통과 기준($6.25 \times 10^{-3}$)은 통과함.
> 5. **수렴성 한계 (Limitation)**: 대형 초기값 $\|z_0\|$ 조건(신규 시드 314159)에서 1/100 궤적 발산이 확인되어 전역 수렴성이 아님을 한계로 명기함.
> 6. **Set A 분리 기록**: 초기 교차표 $45/6/0/49$ 및 $\rho=0.9056$ 수치는 어떤 실행으로도 재현 불가한 출처 불명 수치로 최종 분류하여 §5.5에 봉인함.

---

## 3. Step 12 대조군 분석 최종 판정 (Path A' Robustness Adjudication)

- **Vanilla DEQ (미제약 MLP $34 \rightarrow 64 \rightarrow 32$)**: 4회 독립 GPU 실행 간 힘 비대칭도 $9.44\% \sim 11.31\%$, 공형 잔차 $R \sim 3.99 \times 10^{-4} \sim 4.99 \times 10^{-4}$, 수렴율 0/100 (잔차 평균 $\sim 5.88 \times 10^{-3}$, $\rho \approx 0.9999$ 한계 안정), 600스텝 종점 표류 산포 대표 실행 91군집 (채널 조건부 92군집).
- **Monotone DEQ (수축 사상 $\|W\|_2 \le 0.9$)**: Banach 수축 정리 및 실측 결과 $N_{\text{basins}} = 1$ (단일 끌개 집착 `(+0.024315, +0.000215)`*; *주: 미봉인 채널 산출 대표 좌표 [seed 7777], 정본 게이트 수치 아님*), 훈련 loss_eq가 Epoch 10부터 $0.438555$에 동결되어 부호 의존 타깃 학습 불가능 입증.
- **최종 감사 판정**: 비트 단위 재현 대신 4회 독립 실행에 대한 **실행 간 강건성(Cross-run Robustness)**을 과학적 증거로 채택하여 Step 12 판정을 공식 종결함.

<!-- END OF FREEZE_PAPER2 -->
