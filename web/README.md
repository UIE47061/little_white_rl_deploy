# 小白手機網頁控制台

網頁與 DDS/RL 控制器整合在同一個部署專案。3D 小白使用 **DDS 回傳的關節角度與 IMU**，固定顯示在畫面中央。機身顯示高度是固定的展示位置，不代表真實離地高度或世界位置。

## 啟動：先接 MuJoCo

沿用已有 torch、numpy、PyYAML、MuJoCo、pygame、Unitree SDK2 Python 與 CycloneDDS 的機器人環境，例如本機的 `unitree_mujoco` conda 環境。Python 需 3.10 以上。

安裝選用的網頁依賴，在 `little_white_rl_deploy` 目錄執行：

若使用本專案的 `wtw-mjlab`／uv 虛擬環境，該環境預設可能沒有 `pip`。使用 uv 直接安裝到目前 Python：

```bash
uv pip install --python "$(python -c 'import sys; print(sys.executable)')" -r web/requirements.txt
```

已安裝 pip 的環境也可以使用：

```bash
python -m pip install -r web/requirements.txt
```

終端機一先啟動現有模擬器：

```bash
cd ../csl_mujoco/simulate_python
python unitree_mujoco.py
```

模擬器目前設定為 `little_white_v3`、domain 1、`lo`。終端機二在 `little_white_rl_deploy` 啟動控制台：

```bash
python little_white_wtw.py lo --web
```

電腦可以開啟 `http://127.0.0.1:8080`。手機連上相同路由器後，開啟 `http://電腦的區域網路IP:8080`。預設 HTTP 監聽 `0.0.0.0:8080`，路由器需允許裝置互通，電腦防火牆需允許此連接埠。

`lo` 是**模擬器與控制器之間的 DDS 介面**，手機透過 HTTP/WebSocket 連線，不需要為了手機而把它改成 Wi-Fi 介面。

自訂監聽位置或連接埠：

```bash
python little_white_wtw.py lo --web --web-host 0.0.0.0 --web-port 8081
```

啟動順序：**取得控制 → Stand → 等待站姿轉換完成 → Move → 操作菇菇頭**。也可以先 Sit，再 Stand。Ctrl+C 關閉控制器。網頁模式不開 pygame 視窗、不需要終端機輸入；省略 `--web` 仍使用現有操作方式。

此版本以可信任的區域網路為使用環境，無雲端帳號或登入系統。實機部署需另外核對 DDS 介面、domain、馬達順序、IMU 座標與停止行為；模擬驗收不代表已完成實機驗證。

## 操作與參數

- 左菇菇頭：前後、左右側移；右菇菇頭：左右轉向。支援雙指，中央 10% 為死區。速度限制取自 YAML；前進與後退採不同限制。
- 放開某個菇菇頭，清零其控制軸，仍保持 Move。一般操控／心跳以 20 Hz 傳送，放開時立即傳送零值。
- Stand／Sit 使用固定關節姿態與 YAML 的轉換時間（預設 3 秒）。Move 使用 RL。每次進入 Move 都清除速度，需重新觸控。
- 身體高度偏移、踏頻與抬腳高度常駐。滑桿拖動時最多每秒更新 10 次，放開時送出最終值；以後端確認值顯示。Stand／Sit 時可預設，Move 時才生效。
- 主畫面提供 TROT／PACE／BOUND／PRONK 四個步態按鈕，直接點選切換，以後端確認的步態高亮顯示。
- 右側「調整參數」展開俯仰、側傾、站距、支撐相比例與恢復預設值。面板內容可捲動，48 px 關閉按鈕固定在頂部並避開螢幕安全區；也能點面板外關閉。切換參數不重設 policy 歷史／步態相位。支援 idle control 的 policy 才會出現原地踏步切換。
- 模型可拖曳旋轉、雙指縮放；「重置視角」恢復預設相機。模型不直接依指令產生行走動畫。
- 一個連線取得控制，其餘連線觀看。切換模式、釋放控制、斷線後，不重播舊菇菇頭輸入。
- 「停止／阻尼」為現有 `stop`：位置增益歸零、速度阻尼開啟，是軟體阻尼停止。

## 斷線保護與程序隔離

Controller 在自己的控制迴圈檢查 **300 ms 操控逾時**。手機鎖屏、切背景、關閉分頁或網頁子程序崩潰後，移動命令會歸零；當時若為 Move，切回 Stand。Sit、Stand、阻尼停止維持原模式。重連後必須重新取得控制與啟動 Move。

DDS 狀態超過 YAML `state_timeout`（預設 **500 ms**）未更新時，優先切阻尼，不能依過期狀態返回站姿。網頁也會凍結最後姿態並標示資料過期。以上為軟體逾時目標，受作業系統排程影響。

FastAPI 在獨立子程序運行。父程序只有一個 policy 與 DDS 發送端；原本 200 Hz 發送與 50 Hz 推論排程保持獨立。橋接執行緒透過有界佇列處理指令及 30 Hz 狀態快照；網路與 JSON 處理不在控制迴圈內。觀看端只傳送最新狀態，慢速連線不累積狀態串流。

## 前端建置與離線使用

`web/dist/` 已包含建置結果、URDF、STL 與第三方授權。**一般運行不需要 Node.js 或外網**。修改前端才需要 Node.js（建議 22 LTS）：

```bash
cd web/frontend
npm ci
npm run build
```

Vite 建置輸出到 `web/dist/`。提交／複製專案時一起保留這個目錄。開發伺服器可用 `npm run dev`，需同時啟動 8080 後端；`/api` 與 `/ws` 會代理到後端。

模型來源：`csl_mujoco/unitree_robots/little_white_v3`。URDF 與 meshes 保持原始相對路徑與比例；來源專案的授權保存在 `public/robot/LICENSE`。Three.js MIT 與 URDFLoader Apache-2.0 授權保存在 `public/licenses/`，建置後亦隨本地資源提供。

## HTTP / WebSocket 介面

- `GET /api/config`：參數上下限、步長、單位、預設值、速度限制、步態、idle 能力、馬達名稱、模型路徑與逾時。
- `WS /ws`：連線先收到 `hello`（包含本連線 ID），之後收到 `state`。瀏覽器需同源。
- 請求類型：`acquire`、`release`、`mode`、`control`、`parameters`、`gait`、`reset`。需要回覆時附整數 `id`，得到 `result`，內含 `ok`、錯誤或實際接受的狀態。
- `control` 同時是心跳，包含 `vx`、`vy`、`yaw_rate` 與目前 `state.epoch`；使用物理單位 m/s、rad/s。只有持有控制權的連線有效。
- `mode` 帶 `mode: "stand" | "sit" | "move" | "stop"`；`parameters` 帶 `values`（例如 `{"frequency":2.8}`）；`gait` 帶 `gait` 名稱。
- `state` 包含實際 `q`、`dq`、`quaternion`（wxyz）、狀態年齡、模式、轉換進度、控制者、epoch、命令設定及錯誤。`q`／`dq` 依 `/api/config.motor_joints` 對應。前端與 URDF 使用 X 前、Y 左、Z 上，四元數只轉換為 Three.js 的 xyzw 格式。
- 模式／控制權改變時 epoch 更新，舊 epoch 的速度封包拒絕。伺服器接收超過 300 ms 的排隊命令拒絕；心跳不能自動取得控制或啟動 Move。
