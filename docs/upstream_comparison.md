# 與 morrisx28 原版的差異

比較日期：2026-10-19。原版來源為 [morrisx28/little_white_rl_deploy](https://github.com/morrisx28/little_white_rl_deploy/tree/2cc9e5167dcd515ad667af6decbc44c64b9df452)，本次實際下載的預設分支 HEAD 為 `2cc9e5167dcd515ad667af6decbc44c64b9df452`。不是拿目前工作區的 Git HEAD 當原版。

原版只有 `little_white_isaaclab.py/.yaml`、`little_white_test.py`、`little_white_him.yaml`、`gui_teleop.py` 及 HIM／Isaac Lab 權重；沒有 MJLab、WTW 或網頁控制台。因此比較行走部署時，以原版 Isaac Lab 入口的實際程式行為為準。

## 保留與恢復的行為

| 項目 | 原版 | 目前 MJLab 行走版 |
|---|---|---|
| 通訊 | Unitree SDK2 DDS，`rt/lowstate`／`rt/lowcmd`，CRC | 相同 topic、訊息與 CRC |
| 預設連線 | domain 1、`lo`；第一個命令列參數指定網卡 | 相同；另支援 `--domain-id` 與 YAML |
| 低階命令 | 關節 q、kp、kd；dq 與前饋 tau 為 0 | 相同 |
| 名義週期 | 5 ms 發送、每 4 次更新 policy（50 Hz） | 5 ms 發送、20 ms policy；按單調時鐘排程 |
| 操作流程 | `stand`、`sit`、`move`、`exit` | 保留；新增 `stop`、`status`、`cmd vx vy yaw` |
| 鍵盤行為 | 每次按鍵增減速度，放開保持 | 本次恢復為預設 `incremental`，步幅均為 0.2 |
| 方向按鍵 | 實際程式 W/S=vx、A/D=vy、Q/E=yaw | 相同；原版面板文字把 A/D 與 Q/E 寫反，目前文字配合實際行為 |
| Space／Esc | 歸零，Esc 不關窗 | 相同；另在失焦、關窗時歸零 |
| 目標關節公式 | default + scale × action | 相同形式，數值依新模型 metadata |

鍵盤上限已集中在 `little_white_mjlab.yaml`，保留使用者現有上限設定，並與 metadata 的訓練範圍取交集。有效前進上限 1.0 m/s、後退 0.2 m/s、側移 0.3 m/s、轉向 0.5 rad/s。原版入口使用 `max_lin=0.8`、`max_ang=0.5`，前後／側移共用對稱線速度上限；新模型的各軸範圍不對稱，不能照搬。需要原本的前進上限時可將 `max_lin` 改成 0.8。

## 因新模型而必要的差異

| 項目 | 原版 Isaac Lab 入口 | 目前 MJLab omni |
|---|---|---|
| 模型 | `pre_train/isaaclab/0320_2/policy.pt`，設定為機器專屬絕對路徑 | `pre_train/mjlab/omni/policy.pt`，相對 YAML 路徑＋SHA256 |
| policy 輸入 | `[1,45]` 單幀 | `[1,20,45]`，20 幀由舊至新，首幀重複初始化 |
| 每幀順序 | 角速度、重力、命令、相對 q、dq、前次 action | 角速度、重力、相對 q、dq、前次 action、命令 |
| dq 縮放 | 程式實際乘 0.05 | 目前採原始 rad/s，scale=1 |
| 角速度／命令縮放 | YAML 有縮放值，但 `step()` 實際未使用 | 目前 scale=1 |
| normalization | 入口不額外處理 | 新模型已內含，入口不重複處理 |
| 關節順序 | 直接以 DDS 索引讀寫 | 明確映射：policy FR/FL/RL/RR；本機 DDS FL/FR/RL/RR |
| 站立預設角度 | 前腿 `[0,.5,-.9]`，後腿 `[0,.7,-.8]` | 四腿 `[0,.95,-1.7]` |
| action scale | hip=.125，其餘=.25 | 全部=.25 |
| 行走 PD | 每腿 kp=`[20,25,20]`、kd=`[.4,.6,.4]` | kp=20、kd=.5，從 metadata 讀取 |
| 坐姿 | 原版配置的坐姿向量 | 小白 V3 現有坐姿，依 policy 順序映射到 DDS |

這些數值不能為了讓程式看起來相同而改回舊值；它們與不同模型的訓練介面相連。原訓練端完整設定尚未提供，因此相對關節位置、原始物理單位和不額外裁切 action 仍屬目前接入假設；短程模擬通過不代表它們已由訓練原始碼確認。

## 實作與其他功能差異

- 原版 OpenCV 面板在執行緒內運作；目前使用 pygame 獨立程序。按鍵增減語意已恢復，視覺外觀不同，另可設定 `keyboard_mode: hold`。
- 原版站／坐使用 tanh 插值及當前量測位置；目前從切換瞬間的位置線性插值 3 秒，使用新模型的 PD 增益。
- 目前等待有效 DDS 狀態才啟動，狀態超過 0.5 秒未更新會切阻尼；進入 move 要先完成 stand。這些不是原版已有的行為。
- 原版入口還有 `chirp`、`plot`、`save` 診斷命令。目前 MJLab 入口沒有恢復這三項，避免把量測工具重寫混入行走模型接入；主要操作命令已列於上表。
- 本機原本另有 MJLab 單幀 48 維入口，這是後續 fork 加入，並不在上述原版 commit。其舊檔已移至 repository 外的本機封存，不隨部署專案提供，不能當作新模型的輸入介面。
- 本機的 WTW 模型、網頁、安裝文件均為後續擴充。此次不刪除既有行走與網頁功能。MJLab 目前借用 `little_white_wtw.Controller` 的 DDS 傳輸、排程、阻尼與關節映射檢查，沒有使用 WTW 觀測或 policy。

## 已完成的驗證範圍

測試程式已從部署 repository 移除。整理前完成的驗證包含模型載入與形狀、關節／感測器順序及 torque limit 檢查；觀測歷史與 policy 輸出一致性；鍵盤增減／保持／歸零／失焦行為；隔離 domain 87、lo 的 MuJoCo 站立、低速行走與阻尼停止。未完成全速度範圍及實機驗收。

```bash
python little_white_mjlab.py --check
python little_white_mjlab.py --check-mapping ../csl_mujoco/unitree_robots/little_white_v3/scene.xml
```
