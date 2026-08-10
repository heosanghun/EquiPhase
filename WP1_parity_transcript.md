[WP-1 완료 보고]

1. 실행 스크립트 해시 (매니페스트 대조: 일치/불일치)
- 해당 없음 (실행이 아닌 코드 전사 WP)

2. 실행 명령어 + exit code
- 해당 없음

3. stdout/stderr 전문 (코드 전사 결과 본문)

### (a) vanilla score network
- 파일: `claude_ala2_phase2_train_gates_sealed.py` (SHA256: 48DE453DA3F66D7FDB6F715304DC00C55AEBCF6C4CC9797862A7CD21A9AC4616)
- 라인 번호: 75-76
- 레이어 구조: `nn.Sequential(nn.Linear(4, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2))` (단, h=64)
- 활성함수: ReLU
- 총 파라미터 수: 4,610

### (b) monotone map
- 파일: `claude_ala2_monotone_r4a_sealed_v2.py` (SHA256: F548E129100FDD7B3B2319C267004AE71F9801B15F94B3660A5E5CB227F60635)
- 라인 번호: 51-58
- 구조: `(1.0 - self.lam) * q + self.lam * m` (여기서 m은 `self.scale * torch.tanh(self.l2(torch.tanh(self.l1(enc(q)))))`)
- 수축 보장 메커니즘: `nn.utils.spectral_norm` 적용 및 스케일링(`scale=0.9`, `lam=0.5`)을 통한 Lipschitz 상한 0.95 보장.
- 총 파라미터 수: ~450 (4x64 + 64x2 선형 레이어 기준)

### (c) EquiPhase Vθ
- 파일: `claude_ala2_phase2_train_gates_sealed.py` (SHA256: 48DE453DA3F66D7FDB6F715304DC00C55AEBCF6C4CC9797862A7CD21A9AC4616)
- 라인 번호: 67-68
- 구조: `nn.Sequential(nn.Linear(4, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))` (단, h=64)
- 활성함수: Tanh
- 총 파라미터 수: 4,545

### (d) Common
- 파일: `claude_ala2_phase2_train_gates_sealed.py`
- 학습률: 1e-3 (라인 25, 101)
- 옵티마이저: `torch.optim.Adam` (라인 101)
- solver 반복 예산: 초기화당 2000 (라인 29 `DESC_STEPS = 2000`, 라인 175 `for _ in range(DESC_STEPS):`)
- 정지 기준/residual 임계값: early stopping 없이 정확히 2000 스텝 반복 (정지 기준 없음)

### (e) η, Δt, m, γ 실제 값
- η (Monotone relaxation factor): 0.5 (`claude_ala2_monotone_r4a_sealed_v2.py` 라인 51 `lam=0.5`)
- Δt (Gradient descent lr/step): 0.05 (`claude_ala2_phase2_train_gates_sealed.py` 라인 28 `DESC_LR = 0.05`)
- m, γ: 해당 평가 스크립트에서는 모델링되지 않음(과감쇠 극한으로 가정).

4. 산출 파일 목록 + 각 해시
- `WP1_parity_transcript.md` (본 문서)

5. 이상 사항
없음
