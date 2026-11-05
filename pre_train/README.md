# 部署模型

`policy.pt` 與 `metadata.json` 必須成對保存；不同觀測格式不能只替換 PT。

| 模型目錄 | 入口 | 輸入 | PD |
|---|---|---|---|
| `mjlab/proprio_v7` | `little_white_mjlab.py` 預設 | 單幀 45 維 | 40 / 0.1 |
| `wtw/wtw_kp40_precision_20261105_1500` | `little_white_wtw.py` 預設 | 70 維 × 30 幀 | 40 / 0.1 |

WTW 預設為精度修正訓練的第 1,500 輪，Kp=40/Kd=0.1；預設 2.5 Hz trot 持續踏步。
20 情境各 4 seeds × 20 秒，18/20 通過嚴格診斷，全部存活；±0.25 rad/s 小幅轉向精度仍不足。
這不是完整 20 seeds 驗收，尚待完整 DDS 動態驗證。結果見模型目錄 `precision_evaluation.json`。
其他步態、頻率、高度與姿態指令尚未重新驗證。

舊模型保留。WTW v8 的原始驗收紀錄保存在其目錄。模型實際路徑由 YAML 決定，來源絕對路徑僅供追溯。

```bash
python little_white_mjlab.py --check
python little_white_wtw.py --check
python little_white_wtw.py --check-mapping ../csl_mujoco/unitree_robots/little_white_v3/scene.xml
```
