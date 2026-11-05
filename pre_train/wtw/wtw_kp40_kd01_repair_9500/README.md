# WTW Kp=40 / Kd=0.1 · repair seed1 · iteration 9500

`little_white_wtw.yaml` 的預設模型，輸入為 70 維 × 30 幀、輸出為 12 個原始 actions。
來源：`walk_these_ways_mjlab/runs/wtw_kp40_kd01_repair_20261104_seed1/model_9500.pt`。
完整 checkpoint／policy SHA256、訓練設定與來源版本記錄在 `metadata.json`。
TorchScript 匯出最大誤差為 0。模型與 metadata 必須成對保存。

預設 trot：2.5 Hz、duty 0.5、swing height 0.06 m；50 Hz 推論、200 Hz PD。
部署速度上限：前進 0.3、後退 0.2、側移 ±0.15 m/s、yaw ±0.5 rad/s。
終端機 `gait trot/pace/bound/pronk` 切換步態；鍵盤支援步態／姿態，搖桿只控制速度。

`acceptance.json` 保存來源 checkpoint 的獨立 MuJoCo 評估摘要：
20 seeds、每回合 20 秒，tracking 7/7、gaits 4/4、switches 1/1，240 回合全部完成。
步態通過不代表所有步態的速度、方向追蹤都達標。pace 時序接近門檻，bound/pronk yaw 誤差仍偏大；
負重、坡面與部分姿態控制未達標，未進行實機驗證。

`deployment_check.json` 是此次接入檢查：128 幀（含四種步態切換）與訓練觀測／關節目標比對，
以及使用本地 `csl_mujoco` 模型和 DDS motor 排列的直接 MuJoCo PD 短測。
原地 trot 與四步態前進各完成 20 秒；這是離線整合測試，未發送 DDS。

## 後續零速度診斷

完整 DDS 測試（MuJoCo 3.8）確認零速度時原 policy 的 roll 峰對峰約 19.2°、
機身高度起伏約 2.6 cm；不經 DDS 的直接模擬也重現相近晃動。基本步態驗收沒有涵蓋這項機身穩定性。
依使用者需求，零速仍由 policy 持續原地踏步，不切換固定站姿，PD 維持 40/0.1。
原 policy 的原地踏步與行走機身穩定性尚未通過更嚴格驗收，正在修正訓練目標。
