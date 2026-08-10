# WP-6 고차원 CV 시스템 데이터셋 확장 조사 보고서

본 보고서는 PI의 조건부 사전 승인 하에 진행된 고차원 복합계(Multi-dimensional Complex System) 확장을 위한 후보 데이터셋 예비 조사 결과입니다. 어떠한 모델 학습이나 평가도 실행되지 않았습니다.

## 1. 후보 1: Chignolin (CLN025) Folding Trajectory
- **출처 URL**: [mdshare (Chignolin)](https://markovmodel.github.io/mdshare/CLN025/)
- **단백질 설명**: 10개의 아미노산으로 이루어진 미니 단백질 (빠른 접힘/풀림 역학)
- **제공 데이터 크기**: 2.9 GB (수백만 프레임의 MD 궤적, 3D 좌표 및 Dihedral angle 포함)
- **라이선스**: CC BY 4.0 (상업적 이용 및 변형 가능, 논문 인용 필수)
- **특징**: 알라닌 다이펩타이드(2차원)를 넘어 10차원 이상의 집단 변수(Collective Variables, CV)를 테스트할 수 있는 표준 벤치마크.

## 2. 후보 2: Villin Headpiece (HP35)
- **출처 URL**: [D. E. Shaw Research (DESRES)](https://www.deshawresearch.com/downloads/download_trajectory_villin.cgi)
- **단백질 설명**: 35개 잔기로 구성된 고속 접힘 단백질
- **제공 데이터 크기**: 약 100 μs 이상의 초장기 시뮬레이션 궤적 (압축 시 약 15 GB)
- **라이선스**: Non-Commercial Research Use Only (학술 연구용 무료, DESRES 라이선스 동의 필요)
- **특징**: 고차원 복합계에서 EquiPhase 모델의 위상적 안정성을 증명하는 데 있어 가장 권위 있는 데이터셋 중 하나.

## 3. 후보 3: BPTI (Bovine Pancreatic Trypsin Inhibitor)
- **출처 URL**: [SimTK (BPTI Fold/Unfold)](https://simtk.org/projects/bpti)
- **단백질 설명**: 58개 잔기, 다중 이황화 결합을 포함한 견고한 단백질
- **제공 데이터 크기**: 1ms 시뮬레이션 궤적 1밀리초 궤적 (수십 GB)
- **라이선스**: MIT 및 GNU GPL (오픈소스 학술 프로젝트용)
- **특징**: 국소적 얽힘(Local frustration)과 다중 메타스테이트(Metastable states)가 혼재하여, EquiPhase 모델이 '가짜 골짜기(Artifacts)'를 얼마나 잘 억제하는지 극한 테스트 가능.

## 종합 의견 (Auditor/PI 참고용)
EquiPhase 논문의 D4' 범위 확장 실험을 위해서는 **후보 1 (Chignolin)**이 가장 적합할 것으로 판단됩니다. 오픈소스 라이선스 구조가 명확하며, 계산 비용이 ICLR 마감 기한 내에 소화 가능한 수준입니다. PI의 최종 승인이 있을 경우, 사전등록 수정안(Amendment #3) 작성 후 즉각적인 다운로드 및 데이터 해시 봉인(WP-0) 절차를 시작할 수 있습니다.
