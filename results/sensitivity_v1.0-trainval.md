# F1 (IMM) 민감도 — v1.0-trainval val split

| 변경 | TTC MAE [s] | 객체 상태 정확도 | 오경보율 |
|---|---:|---:|---:|
| 기본값 | 0.967 | 67.3% | 0.6% |
| {"imm": {"p_stay": 0.9}} | 0.963 | 70.1% | 0.5% |
| {"imm": {"p_stay": 0.99}} | 0.984 | 66.5% | 0.7% |
| {"imm": {"mu_stationary0": 0.4}} | 0.968 | 66.3% | 0.7% |
| {"imm": {"mu_stationary0": 0.8}} | 0.978 | 68.4% | 0.5% |
| {"imm": {"pseudo_rel": 0.02}} | 0.968 | 67.3% | 0.6% |
| {"imm": {"pseudo_rel": 0.1}} | 0.966 | 67.2% | 0.6% |
| {"imm": {"sigma_a_stationary": 0.1}} | 0.968 | 67.3% | 0.6% |
| {"imm": {"sigma_a_stationary": 1.0}} | 0.965 | 67.3% | 0.6% |
| {"kalman": {"init_sigma_v": 3.0}} | 0.995 | 68.4% | 0.4% |
| {"kalman": {"init_sigma_v": 8.0}} | 0.957 | 66.5% | 0.8% |
| {"kalman": {"sigma_a": {"vehicle": 1.0, "pedestrian": 1.0}}} | 0.987 | 66.8% | 0.6% |
| {"kalman": {"sigma_a": {"vehicle": 3.0, "pedestrian": 1.0}}} | 0.958 | 67.2% | 0.6% |
