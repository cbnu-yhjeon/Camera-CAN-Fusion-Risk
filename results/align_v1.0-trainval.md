# CAN ↔ 카메라 시간 정렬 (v1.0-trainval)

- scene 850개 중 CAN 없음 15개 (devkit blacklist + 파일 결손)
- 주행 중 scene 723개의 lag 중앙값 +0.08 s, IQR [+0.05, +0.12] s, |lag| > 0.1 s 비율 41.5%
- 속도 RMSE (보간 그대로 / lag 보정 후) 중앙값: 0.143 / 0.139 m/s
- CAN 이 scene 일부만 덮는 scene: 0개
