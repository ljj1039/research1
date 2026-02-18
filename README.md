# Linkage Mechanism Multi-Objective Optimization Tool

4절 링크 메커니즘의 다목적 최적화를 위한 Python 도구.

## 구조

```
mechanism_optimizer/
    linkage.py        # 4절 링크 운동학 (위치, 속도, 전달각)
    objectives.py     # 목적함수 (경로오차, 전달각, 기계적이득, 곡선매끄러움)
    optimizer.py      # NSGA-II 기반 다목적 최적화
    visualization.py  # 메커니즘 및 결과 시각화
main.py               # 실행 예제
```

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
python main.py
```

## 기능

- **운동학 해석**: 4절 링크의 위치/속도 해석, Grashof 조건 판별
- **목적함수**: 경로오차, 전달각 편차, 기계적이득 균일성, 커플러 곡선 매끄러움
- **다목적 최적화**: NSGA-II (pymoo) 기반 Pareto 최적 해 탐색
- **시각화**: 메커니즘 형상, 커플러 곡선, Pareto front, 종합 분석 차트
