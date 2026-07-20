# UR5 + Robotiq 2F-85: Target-Aware Pick Task

**Isaac Lab 5.1.0 + RSL-RL PPO**를 이용한 UR5 로봇의 target-aware pick 강화학습.

## 🎯 프로젝트 목표

- 빨강/파랑 큐브 중 지정된 색 큐브 pick
- 카메라 기반 color-conditioned RL로 확장
- 최종: 지정 색 큐브 pick → 다른 큐브 위 stack
- Target: PRESM2026 (2026년 11월, 제주)

## 🤖 하드웨어 & 시뮬레이션

- **로봇**: UR5 + Robotiq 2F-85 gripper
- **DoF**: 5 (wrist_3_joint = PhysicsFixedJoint, top-down 강제)
- **시뮬레이션**: Isaac Sim 5.1, Isaac Lab 5.1.0
- **RL**: RSL-RL PPO (v2 NaN 패치 적용)
- **GPU**: RTX 5080

## 📊 진행 상황

### ✅ 완료
- Baseline: 큐브 1개 pick 성공 (Franka lift 예제 UR5 이식)
- Wrist twist 해결 (wrist_1/2 deviation penalty)
- Distractor 추가 (blue cube) - target만 pick 성공
- Target ID 랜덤화 (M1 옵션 C, 학습 진행 중)

### 🚧 진행 중
- **M1 옵션 C**: Target ID (red or blue) 매 리셋마다 랜덤, 큐브 위치 고정
  - 이전 시도 (M1 v1) 발산 → 원인: 큐브 위치 랜덤화 + reward scale
  - 옵션 C: 큐브 위치 고정으로 학습 안정성 확보

### 📅 계획
- Vision RL (RGB 카메라 + CNN encoder)
- Color-conditioned observation
- Stack 태스크 확장

## 🗂 파일 구조

- `src/joint_pos_env_cfg.py`: UR5 태스크 config (활성)
- `reference/`: Base Isaac Lab 파일들 (수정 대상 아님, 참조용)
- `scripts/auto_train_lift_30k.sh`: 자동 재시작 학습 스크립트
- `logs/`: 학습 로그 요약

## ⚠️ 알려진 이슈

1. **Wrist_3 fixed의 한계**: 5DoF라 top-down만 가능, 옆에서 잡기 못 함
2. **Isaac Sim mutex 크래시**: 3~4시간마다 발생 가능 → auto-restart로 대응
3. **M1 v1 발산**: 큐브 위치 + target 랜덤화 조합에서 학습 불안정 (해결됨: 옵션 C)

## 🔧 실행 방법

```bash
# 학습 시작 (auto-restart 포함)
bash scripts/auto_train_lift_30k.sh

# 평가 (GUI)
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Lift-Cube-UR5-Robotiq-Play-v0 \
    --num_envs 16 \
    --checkpoint {체크포인트 경로}
```

## 📝 참고

- USD 파일: `/home/choi/ur5_robotiq_2f85.usd` (wrist_3 = PhysicsFixedJoint at π)
- PPO 패치: `reference/rsl_rl_ppo_patched.py` (nan_to_num + skip step on non-finite grad)
