# 실험 히스토리

## 7월 초 ~ 7월 중순: Pick 태스크 시행착오
- Custom pick task 코드 여러 번 실패
- Fake reward hacking (20% success인데 실제 pick 없음)
- Value function 발산 반복
- 5/21 lift task working config 재발견

## 7/16-17: Franka Lift 예제 UR5 이식 성공
- Setup: gripper_drive stiffness=11.25, damping=0.1
- Wrist_3 USD FixedJoint 유지 (top-down 강제)
- Wrist_1, wrist_2 deviation penalty 추가 (twist 방지)
- 결과: 큐브 1개 pick 성공, twist 없음

## 7/19: Distractor 추가 (B1)
- Blue cube (distractor) 고정 위치에 추가
- Target(red)만 policy가 관측
- 결과: Distractor 무시하고 red만 pick 성공

## 7/20 아침: Target-aware (M1 v1) - 실패
- Target ID 매 리셋마다 랜덤
- 큐브 두 개 모두 랜덤 위치
- Target-aware reward 함수 4개
- 결과: 5시간 학습 후 policy 발산 (action noise std 4.22, value_function loss inf)
- 원인: 큐브 위치 + target 이중 랜덤화가 학습 불안정 유발

## 7/20 오후: M1 옵션 C (진행 중)
- 큐브 위치 고정 (red: [0.5, 0.0, 0.055], blue: [0.5, 0.25, 0.055])
- Target ID만 랜덤 (0=red or 1=blue)
- 5분 검증 학습에서 value_function loss 안정 (0.18)
- Fresh 학습 시작 (30000 iter, ETA 14시간)
