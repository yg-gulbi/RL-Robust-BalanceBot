# RL-Robust-BalanceBot

> **ROS 2 및 MuJoCo/Gazebo 기반 2륜 가변 링크 자립 밸런싱 로봇 시뮬레이션**  
> *스케이트파크 험지 주파, 이득 스케줄링 LQR 동적 평형 제어, 점프 착지 충격 흡수, 실시간 3D 원격 조종 지원*

[![ROS 2](https://img.shields.io/badge/ROS%202-Jazzy%20Jalisco-blue.svg)](https://docs.ros.org/en/jazzy/)
[![Physics Engine](https://img.shields.io/badge/Physics-MuJoCo%203.14%20%7C%20Gazebo%20Harmonic-orange.svg)](https://mujoco.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-407%20Passed-brightgreen.svg)]()

---

## 📌 프로젝트 소개 (Project Overview)

**RL-Robust-BalanceBot**은 극단적인 스케이트파크 환경(평지, 연속 둔덕, 거친 요철, 12° 경사로, 점프대)을 안정적으로 주파할 수 있도록 설계된 **6자유도(DOF) 액티브 레그(Active-Leg) 기반 2륜 자립 밸런싱 로봇** 시뮬레이션 시스템입니다.

다물체 동역학 기반의 해석적 연속시간 **이득 스케줄링 LQR-I 제어기**, 2자유도 시상면 가상 모델 제어(**VMC, Virtual Model Control**), 공중 자유낙하 및 착지 충격 분산을 위한 **5단계 비행 유한 상태 머신(FSM)**을 유기적으로 결합하여 강력한 외란 극복 및 자세 안정성을 달성했습니다.

![평가 대시보드](docs/evaluation_dashboard.png)

### 핵심 기능
- **6-DOF 하이브리드 기구학/동역학 모델**: 2개 연속 구동 휠 + 4개 피치 회전 관절(골반/무릎).
- **이득 스케줄링 LQR-I 균형 제어**: 다리 길이 변화($L \in [0.18, 0.38]\,\text{m}$)에 맞춰 연속 리카티 방정식(ARE)을 기반으로 최적 피드백 게인을 100 Hz로 실시간 스케줄링.
- **2-DOF 시상면 다리 가상 모델 제어 (VMC)**: 차체 높이 및 앞/뒤 바퀴 오프셋을 자유롭게 조절하고, 오프셋에 따른 동적 평형각($\theta_{eq} = \arcsin(\Delta x / L_{eff})$)을 자동 보정.
- **5단계 비행 및 충격 흡수 FSM**: 자유낙하 감지($||\mathbf{a}|| < 2.5\,\text{m/s}^2$), 공중 휠 회전 폭주 방지(토크 차단), 착지 충격 감지($||\mathbf{a}|| > 15\,\text{m/s}^2$) 및 댐핑 컴플라이언스 전환.
- **복합 센서 시뮬레이션**: 200 Hz 6축 IMU, 20 Hz 360° 2D LiDAR, 30 Hz RGB-D 뎁스 카메라.
- **이중 물리 엔진 환경 지원**:
  - **MuJoCo (3.14)**: 지연 없는 실시간 3D 뷰어와 키보드/플로팅 무선 조종기 UI를 갖춘 고속 인터랙티브 환경.
  - **ROS 2 Jazzy + Gazebo Sim (Harmonic)**: `ros_gz_bridge`를 통한 정규 ROS 2 노드/토픽 생태계 연동.

---

## 🏗 시스템 아키텍처 (System Architecture)

```
       [ 2D 라이다 (20 Hz) ]      [ RGB-D 뎁스 카메라 (30 Hz) ]
                  \                      /
             +--------------------------------+
             |       상체 베이스 (12 kg)       |
             |       IMU 센서 (200 Hz)        |
             +---------------+----------------+
                    /                 \
         [ 좌측 힙 관절 ]            [ 우측 힙 관절 ]   (Pitch 회전, ±1.2 rad, 최대 50 N·m)
                 |                         |
           허벅지 링크 (0.2m)        허벅지 링크 (0.2m)
                 |                         |
         [ 좌측 무릎 관절 ]          [ 우측 무릎 관절 ]  (Pitch 회전, 0~2.3 rad, 최대 50 N·m)
                 |                         |
            종아리 링크 (0.2m)        종아리 링크 (0.2m)
                 |                         |
         [ 좌측 구동 바퀴 ]          [ 우측 구동 바퀴 ]  (연속 회전, 반경 0.1m, 최대 25 N·m)
            (윤간 거리 Track Width = 0.45m)
```

상세한 수식 유도 및 제어기 설계 명세는 [ARCHITECTURE.md](ARCHITECTURE.md)를 참고하세요.

---

## 🚀 시작하기 & 설치 방법 (Quick Start)

### 권장 환경
- **OS**: Ubuntu 24.04 LTS (Noble Numbat)
- **ROS 버전**: ROS 2 Jazzy Jalisco
- **Python**: 3.12 이상
- **물리 엔진 라이브러리**: MuJoCo 3.14+ (`pip install mujoco glfw`)

```bash
# 1. 저장소 클론
git clone https://github.com/yg-gulbi/RL-Robust-BalanceBot.git
cd RL-Robust-BalanceBot

# 2. 필수 의존성 설치
pip install mujoco glfw numpy scipy matplotlib

# 3. ROS 2 워크스페이스 빌드 (ROS 2 패키지 사용 시)
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

---

## 🎮 시뮬레이터 실행 및 조종 방법 (Simulation & Teleop)

### 1. MuJoCo 실시간 3D 시뮬레이터 (추천 환경)

별도의 복잡한 브릿지 설정 없이, 한 줄의 명령어로 즉시 실시간 3D 뷰어 창과 미니 무선 조종기 UI가 실행됩니다:

```bash
python3 run_mujoco_sim.py
```

#### 키보드 조작 가이드 (3D 뷰어 화면 또는 조종기 창에서 직접 입력)
| 키 | 조작 동작 | 기능 설명 |
| :---: | :--- | :--- |
| **`↑` / `W`** | **전진 가속** | 키를 누르고 있는 동안 지속해서 매끄럽게 전진 주행 |
| **`↓` / `S`** | **후진 가속** | 뒤로 후진 주행 |
| **`←` / `A`** | **좌회전** | 차동 구동(Differential Drive)을 통한 부드러운 좌선회 |
| **`→` / `D`** | **우회전** | 차동 구동을 통한 부드러운 우선회 |
| **`Space`** | **긴급 제동 / 정지** | 즉시 제동 후 그 자리에 멈춰 서서 균형 유지(Station-Keeping) |
| **`P`** | **40 N 외란 밀치기** | 로봇에 40 N의 충격을 가해 자세 복원력(Push Recovery) 검증 |
| **`R`** | **자세 초기화** | 시작 직립 스탠스로 즉시 리셋 |
| **`M`** | **조종 모드 전환** | 자동 정지 모드(손 떼면 감속 정지) ↔ 크루즈 모드(속도 유지) 전환 |

---

### 2. ROS 2 + Gazebo Sim (Harmonic)

정규 ROS 2 토픽/노드 파이프라인으로 시뮬레이터를 가동하려면:

```bash
# [터미널 1] 가제보 스케이트파크 월드 및 제어 노드 실행
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch balancebot_description gazebo.launch.py

# [터미널 2] ROS 2 키보드 원격 조종 노드 실행
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run balancebot_controller teleop_node
```

---

## 📊 성능 평가 지표 및 벤치마크 (Benchmarks)

총 407개의 테스트 케이스(물리 한계 시험, 임펄스 외란, 센서 노이즈)를 통과하며 성능을 검증받았습니다:

| 평가 항목 | 목표 성능 기준 | 실제 측정값 | 통과 여부 |
| :--- | :---: | :---: | :---: |
| **평지 정상상태 피치각 오차** ($|\theta|$) | $< 5.0^\circ$ | **$1.84^\circ$** | ✅ 합격 |
| **속도 추종 오차** ($J_v$) | $< 0.15\,\text{m/s}$ | **$0.062\,\text{m/s}$** | ✅ 합격 |
| **40 N 강한 외란 충격 복원 시간** | $< 1.5\,\text{초}$ | **$0.82\,\text{초}$** | ✅ 합격 |
| **차체 최저 지상고 (Ground Clearance)** | $> 0.05\,\text{m}$ | **$0.091\,\text{m}$** | ✅ 합격 |
| **점프대 착지 충격 균형 복원** | 전복 없음 ($|\theta| < 25^\circ$) | **최대 $4.63^\circ$ 피치** | ✅ 합격 |

자동화 검증 스위트 실행:
```bash
pytest src/balancebot_evaluator/test/
```

---

## 🛠️ 설계 배경: 사전 대화형 툴('자비스') 협업 과정

AI 코딩 에이전트에게 구현을 맡기기 전, 사용자가 직접 만든 시스템 블록다이어그램 프로그램(**'자비스(Jarvis)'**)을 통해 요구사항을 구체화했습니다.

```
프로젝트 아이디어 입력
        ↓
대화 기록 + 현재 설계 상태를 AI에 전달
        ↓
AI가 분석 후 다음 질문·선택지·문서 내용 반환
        ↓
화면에 설계 단계·구성도·블록다이어그램 실시간 생성 및 초안 갱신
        ↓
사용자가 다이어그램을 보며 선택지 선택 또는 피드백 입력
        ↓
설계 반복 (Loop)
        ↓
저장 / 자동 동기화로 실제 파일 반영
        ↓
AI 코딩 에이전트에 최종 전달 (ai_handoff)
```

---

## ⚠️ 한계 및 실패 원인 분석

### 1. ROS 2 + 가제보(Gazebo) 밸런싱 실패
MuJoCo 환경에서는 테스트를 통과했으나, **ROS 2 + Gazebo 실제 시뮬레이션 환경에서는 로봇이 균형을 잡지 못하고 쓰러졌습니다.** 주요 원인은 다음과 같습니다:

- **바닥 마찰과 모터 반응 지연**:
  - 로봇이 살짝 기울었을 때 바퀴를 미세하게 굴려 균형을 잡아야 하는데, 바닥 마찰력 때문에 작은 힘으로는 바퀴가 바로 구르지 못했습니다.
  - 힘이 계속 누적되다가 한 번에 확 풀리면서 바퀴가 헛돌고 로봇이 뒤로 넘어졌습니다.
- **다리와 바퀴의 간섭**:
  - 다리를 굽히거나 펴면서 생기는 힘이 몸체에 전해져, 바퀴가 균형을 잡으려는 동작을 방해했습니다.
- **기존 수학 공식(LQR 제어)의 한계**:
  - 기존 제어 방식은 로봇이 거의 똑바로 서 있을 때만 잘 작동하는 단순화된 공식입니다.
  - 가제보에서 스폰되거나 둔덕을 넘을 때처럼 로봇이 크게 삐끗하면 공식을 벗어나 복구하지 못했습니다.

### 2. AI 에이전트 자동 개발의 한계
- **설계와 실제 구현 사이의 간극**:
  - 초반에 다이어그램과 문서를 아무리 꼼꼼히 짜서 넘겨줘도, 복잡한 물리 시뮬레이션 환경에서 **AI 에이전트가 단번에 목표(완벽한 자립 주행)에 도달하는 것에는 분명한 한계**가 있었습니다.
  - 단위 테스트 통과에만 매몰되다 보니, 가제보의 실제 물리 접촉이나 통신 지연 같은 현실적인 변수를 알아서 해결하지 못했습니다.

---

## 🧠 왜 강화학습(RL)이 필요한가?

초기 블록다이어그램 설계에서 강화학습을 고려했던 이유는 위의 한계를 극복하기 위해서입니다:

- **원래 의도**: 바닥 마찰, 충격, 다리 움직임에 따른 복잡한 반작용을 수식으로 일일이 계산하기 어렵기 때문에, AI가 수많은 시행착오를 겪으며 스스로 요령을 터득하는 강화학습(PPO)이 필요했습니다.
- **누락된 이유**: 초기 구현 단계에서 기본 수식 제어(LQR)로 테스트를 통과시키는 데 집중하다 보니, 실제 강화학습 훈련 환경 구축이 뒤로 밀렸습니다.
- **해결 방향**: 기존 수식 제어기가 기본적인 힘을 주고, 미끄러짐이나 충격 같은 예측하기 힘든 부분만 강화학습 AI가 실시간으로 보정해 주는 방식(잔차 강화학습)으로 가야 가제보와 실물에서도 안정적으로 균형을 잡을 수 있습니다.

---

## 📂 디렉토리 구조 (Directory Structure)

```
.
├── run_mujoco_sim.py             # 실시간 3D MuJoCo 시뮬레이터 및 조종기 UI 러너
├── README.md                     # 프로젝트 전체 안내서 (현재 문서)
├── ARCHITECTURE.md               # 시스템 아키텍처 및 수학적 동역학 모델 명세서
├── TEST_READY.md                 # 테스트 실행 및 검증 가이드
├── docs/                         # 시연 동영상, 대시보드 그래프 및 평가 점수표
│   ├── balancebot_demo.mp4       # 실제 주행 및 장애물 극복 30 FPS 데모 영상
│   ├── evaluation_dashboard.png  # 속도/피치 추종 및 위상 공간 그래프
│   └── scorecard.csv             # 정량 평가 결과 CSV
└── src/
    ├── balancebot_description/    # URDF/Xacro, MJCF 스케이트파크 월드 및 센서 플러그인
    ├── balancebot_controller/     # LQR 밸런싱, VMC 다리 기구학, FSM 상태 관측기 노드
    └── balancebot_evaluator/      # 자동 평가 테스트 하네스 및 스코어카드 산출기
```

---

## 📜 라이선스 (License)

본 프로젝트는 [Apache License, Version 2.0](LICENSE) 라이선스를 따릅니다.
