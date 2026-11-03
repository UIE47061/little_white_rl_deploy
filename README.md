# little_white_rl_deploy

小白 V3 行走 policy 部署：透過 Unitree DDS 讀取關節／IMU，執行 TorchScript，送出關節目標與 PD 增益。

| 版本 | 入口 | 模型 | 控制方式 |
|---|---|---|---|
| MJLab Proprio V7 | `little_white_mjlab.py` | `pre_train/mjlab/proprio_v7/` | 鍵盤、搖桿、終端機 |
| WTW v8 | `little_white_wtw.py` | `pre_train/wtw/wtw_slippery_trot_v8/` | 鍵盤、搖桿、終端機、手機網頁 |

## 安裝

需要 Linux、Python 3.10 以上、Unitree SDK2 Python／CycloneDDS。沿用已準備好的機器人環境，在專案目錄安裝：

```bash
# 鍵盤與模型推論依賴
python -m pip install -r requirements/keyboard.txt
# 使用 WTW 手機網頁時另外安裝
python -m pip install -r requirements/web.txt
```

沒有 pip 的 uv 環境，可改用 `uv pip install --python "$(python -c 'import sys; print(sys.executable)')" -r requirements/keyboard.txt`。
完整環境設定見 [安裝說明](docs/installation.md)。

## 啟動模擬器

`csl_mujoco` 是獨立專案，預期放在本專案旁邊，使用 `little_white_v3`、DDS domain 1、介面 `lo`。另一個終端機執行：

```bash
cd ../csl_mujoco/simulate_python
python unitree_mujoco.py
```

接著選擇一個控制器執行。同一個機器人／DDS domain 使用一個控制器。

## MJLab 行走

目前使用本機 `proprio45_v7_load_balance_seed50/model_2999.pt`，PD 為 **40 / 0.1**，
控制頻率 50 Hz。單幀 45 維觀測：角速度 3、重力方向 3、相對關節角度 12、
關節角速度 12、前次 action 12、速度指令 3；不使用機身線速度、歷史堆疊或高度掃描。
觀測正規化已包在模型內。舊模型留存，但不再由預設設定載入。

程式依原版 `little_white_isaaclab.py` 的單一 `Controller` 架構整理：
`Init()` 建立 DDS、`LowStateMessageHandler()` 更新狀態、
`stand()/sit()/move()` 處理模式、`step()` 執行 policy、
`LowCmdWrite()` 定時發送。模型載入、觀測與關節映射均放在
`little_white_mjlab.py`，不依賴 WTW 或另外的 runtime 模組。
PD、預設姿態與 action scale 明列在 YAML，啟動時與訓練 metadata 核對。

使用已備妥的 Python 3.10 環境（若目前在 mjlab venv，先 `deactivate`）：

```bash
conda activate unitree_mujoco
cd /home/csl/U1_ws/little_white_rl_deploy
```

```bash
python little_white_mjlab.py --check
python little_white_mjlab.py lo
# 純終端機
python little_white_mjlab.py lo --no-gui
```

此模型只需要 `rt/lowstate` 的關節角度、關節速度、IMU 四元數與角速度，
不訂閱 `rt/sportmodestate`，也不需要線速度估計。關節／IMU 狀態過期時切換阻尼。

已檢查匯出前後 actor 輸出一致與 MuJoCo 關節映射。訓練環境的固定命令評估顯示
相較 V6，V7 後腿接地與承重更平均，後退／側移／左轉改善，但直行速度稍慢。
平地固定命令的推擾／噪聲測試未跌倒；此版本用於 sim2sim 測試，尚未完成實機驗證。

重新匯出與離線測試：

```bash
cd ../mjlab
uv run --no-sync ../little_white_rl_deploy/export_mjlab_policy.py \
  --checkpoint logs/rsl_rl/little_white_v3_proprio_v7_easy/2026-11-03_13-59-18_proprio45_v7_load_balance_seed50/model_2999.pt \
  --output ../little_white_rl_deploy/pre_train/mjlab/proprio_v7
# 匯出會產生新的 SHA256，需同步填入 little_white_mjlab.yaml。
cd ../little_white_rl_deploy
python little_white_mjlab.py --check-mapping ../csl_mujoco/unitree_robots/little_white_v3/scene.xml
```

在終端機輸入 `stand`，等 3 秒後輸入 `move`，再點選鍵盤面板：

- W／S：增減前後速度。
- A／D：增減側移速度。
- Q／E：增減轉向速度。
- Space／Esc：速度歸零；Esc 不關閉面板。

MJLab 使用原版 `gui_teleop.py` 的 OpenCV 鍵盤面板，每次增減 0.2，
**放開按鍵會保留速度**；Space／Esc 歸零。此面板不支援 hold 模式。
若 shell 為訓練設定了 `QT_QPA_PLATFORM=offscreen`，有 `DISPLAY` 時面板會自動
改用 OpenCV 附帶的 `xcb`，避免缺少 offscreen 插件造成崩潰。
速度上限、線速度／角速度步幅也在此 YAML 調整；上限會受模型訓練範圍限制。

終端機支援 `stand`、`sit`、`move`、`stop`、`status`、`exit`，以及 `cmd vx vy yaw`。
例如 `cmd 0.2 0 0` 前進、`cmd 0 0 0` 零速度；`stop` 立即切阻尼。

模型輸入是 20 幀 × 45 維：角速度、重力、相對關節角度、關節角速度、前次 action、速度指令。normalization 已內含。
目前已通過短程 MuJoCo＋DDS 行走／停止驗證；原訓練端完整前處理設定仍待核對，尚未完成全速度範圍及實機驗收。
與原始 repository 的差異見 [原版對照](docs/upstream_comparison.md)。

## WTW 行走與手機網頁

```bash
python little_white_wtw.py --check
python little_white_wtw.py lo
python little_white_wtw.py lo --no-gui
# 使用手機網頁操作
python little_white_wtw.py lo --web
```

WTW 鍵盤為按住 W/S、A/D、Q/E 移動，放開歸零；終端機先 `stand`，等待轉換完成，再 `move`。
終端機另支援 `preset slippery_trot`、`preset trot`、`gait trot`、`cmd frequency 2.8`、`status`、`stop`、`exit`。
兩個 preset 都使用目前載入的 v8 模型，改變步態參數。

網頁開啟 `http://127.0.0.1:8080`；手機使用同一區域網路上的 `http://電腦IP:8080`。
操作順序為 **取得控制 → Stand → 等待完成 → Move**。
網頁控制台目前使用 WTW 入口，詳見 [網頁說明](web/README.md)。

WTW v8 已通過部分模擬直行／停止測試，部分左轉仍未通過整體驗收；模型評估紀錄保存在對應模型目錄。

## 搖桿控制（MJLab／WTW 共用）

使用 `requirements/keyboard.txt` 的 pygame 依賴，不開啟鍵盤視窗。接上手把後執行：

```bash
python headless_teleop.py --list
# 啟動 MuJoCo 後，以下擇一
python little_white_mjlab.py lo --gamepad
python little_white_wtw.py lo --gamepad
```

終端機輸入 `stand`，等 3 秒，再輸入 `move`，將搖桿置中後即可操作。
左搖桿控制前後／側移，右搖桿控制轉向，回到中心即送出零速度。
按鈕 0（常見為 A／叉）讓速度指令歸零；放開按鈕並置中才恢復。
終端機 `stop` 切換阻尼。WTW 保留原本速度漸變，零速度指令不代表瞬間停止。
斷線、輸入程序停止或超過 0.25 秒未更新時送出零速度；重新連線後需置中。

各版本 YAML 的 `gamepad` 區塊可調整裝置、軸編號、方向、按鈕與死區。
預設前後 axis 1、側移 axis 0、轉向 axis 3；不同手把可能不同。
用 `python headless_teleop.py --device 0` 查看原始軸與按鈕編號，Ctrl+C 結束。
速度上限使用 YAML 的 `max_lin`、`max_backward`、`max_lateral`、`max_ang`，並受模型範圍限制。
手把只控制速度，不調高度／步態或切換站立模式；手把模式停用終端機速度指令。
WTW 的 `--gamepad` 與 `--web` 不可同時使用；未加 `--gamepad` 維持原本鍵盤操作。

## 模型與通訊設定

兩個入口都支援 `--config 設定檔.yaml`、`--domain-id 數字` 與網卡名稱參數。
模型相對路徑以 YAML 所在目錄為準；policy 與 metadata 成對保存。
`lo` 用於本機模擬，實機網卡與 DDS domain 需依實際連線設定。

```bash
# 無 DDS 的模型／關節映射檢查
python little_white_mjlab.py --check-mapping ../csl_mujoco/unitree_robots/little_white_v3/scene.xml
python little_white_wtw.py --check-mapping ../csl_mujoco/unitree_robots/little_white_v3/scene.xml
```

## 目錄

```text
little_white_rl_deploy/
├── little_white_mjlab.py / .yaml   # MJLab 主程式與設定
├── export_mjlab_policy.py          # 訓練環境用的模型匯出工具
├── gui_teleop.py                  # 原版 OpenCV 鍵盤面板，補上命令重置與各軸限速
├── headless_teleop.py             # 共用搖桿與裝置檢查
├── little_white_wtw.py / .yaml    # WTW 推論與共用 DDS Controller
├── wtw_commands.py               # WTW 指令與步態
├── wtw_motion.py                 # WTW 速度限制
├── wtw_teleop.py                  # WTW 鍵盤
├── pre_train/                    # 目前使用的兩顆 policy、metadata、模型說明
├── requirements/                 # base／keyboard／web 依賴
├── web/                          # WTW 網頁後端、前端原始碼與離線建置結果
└── docs/                         # 安裝、架構、原版對照
```

MJLab 與 WTW 各自使用獨立 Controller；鍵盤與搖桿工具仍分檔保存。
`web/dist/` 是網頁執行必要的離線資源；`web/frontend/` 用來修改與重建網頁。
前端建置方式為 `cd web/frontend && npm ci && npm run build`，建置後一併提交 `web/dist/`。

## Git

提交程式、YAML、兩顆 policy／metadata、網頁原始碼／建置結果與第三方授權。
測試、舊權重、備份、node_modules 和執行快取不在目前部署內容中。

```bash
git status --short
git diff --check
git add -A
git diff --cached --stat
git commit -m "Clean deployment layout and add MJLab omni walking"
git remote -v
# 確認遠端與分支後再 push
```
