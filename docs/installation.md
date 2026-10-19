# 安裝與環境

## 使用已存在的機器人環境

目前工作區的 `wtw-mjlab`／uv 環境已包含 policy、MuJoCo 與 DDS 所需套件。啟用原有環境後，依需求選一組安裝：

| 需求 | 依賴檔 |
|---|---|
| 推論檢查／終端機控制 | `requirements.txt` 或 `requirements/base.txt` |
| 手機網頁控制 | `requirements/web.txt` |
| pygame 鍵盤面板 | `requirements/keyboard.txt` |

在專案根目錄執行，例如：

```bash
uv pip install --python "$(python -c 'import sys; print(sys.executable)')" -r requirements/web.txt
```

uv 建立的環境可能沒有 `pip`；`uv pip` 不需要在目標環境安裝 pip。若環境已有 pip，也可以使用 `python -m pip install -r requirements/web.txt`。不要只執行另一個環境的 `pip`，以免依賴裝到錯誤位置。

既有 `web/requirements.txt` 仍保留，供已具備基礎環境的人單獨安裝網頁依賴。測試依賴不納入部署專案。

## 建立全新環境

已測試的執行環境包括 Linux x86_64、Python 3.10／3.13、PyTorch 2.11.0。以下以 Python 3.13、uv 與 CPU 推論示範，在本專案根目錄執行：

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements/web.txt
python little_white_wtw.py --check
```

`--check` 只載入 YAML、metadata 與 TorchScript policy，不建立 DDS、不控制馬達，也不需要桌面。基礎安裝不包含網頁選用套件時，亦可執行這個檢查。

## Unitree DDS

實際收發 DDS 需要另行安裝 [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python)。SDK 與 CycloneDDS 涉及系統函式庫和平台設定，未硬塞進通用 requirements；請沿用已驗證的機器人環境，或依 SDK 的安裝說明處理 CycloneDDS。

假設 SDK 已下載到本 repository 的同層資料夾，且 CycloneDDS 已依上游文件安裝：

```bash
uv pip install --python "$(python -c 'import sys; print(sys.executable)')" -e ../unitree_sdk2_python
python -c "from unitree_sdk2py.core.channel import ChannelFactoryInitialize; print('SDK import OK')"
```

若出現 `Could not locate cyclonedds`，依 SDK README 編譯／安裝相容的 CycloneDDS，並設定 `CYCLONEDDS_HOME` 指向其安裝目錄，再重試 SDK 安裝。

## MuJoCo 模擬

`csl_mujoco` 是獨立的模擬器專案，沒有包含在本 repository。使用具有 `little_white_v3` 模型與對應 DDS bridge 的版本，放在本 repository 同層：

```text
workspace/
├── little_white_rl_deploy/
├── csl_mujoco/
└── unitree_sdk2_python/      # 若由原始碼安裝 SDK
```

模擬器環境需要 MuJoCo、pygame 與 DDS；請依 `csl_mujoco` 的安裝說明準備，部署依賴不包含模擬器原始碼與 DDS 系統函式庫。

核對 `csl_mujoco/simulate_python/config.py`：

```python
ROBOT = "little_white_v3"
DOMAIN_ID = 1
INTERFACE = "lo"
```

先啟動模擬器，再啟動 `python little_white_wtw.py lo --web`。CLI 的 `--domain-id` 與介面引數只覆寫目前啟動，不修改 YAML。

## 前端

`web/dist/` 已隨 repository 提供，一般啟動不需要 npm。修改 TypeScript／CSS 或模型資源時才需建置，建議使用 Node.js 22 LTS：

```bash
cd web/frontend
npm ci
npm run build
```

前端原始資源放在 `web/frontend/public/`，建置後複製到 `web/dist/`；請一起提交建置結果，維持離線部署能力。

## 本機設定

預設設定保留在 `little_white_wtw.yaml`。需要不同 IP 介面或模型選擇時，可複製成同目錄的 `robot.local.yaml`，再使用：

```bash
python little_white_wtw.py --config robot.local.yaml --web
```

`*.local.yaml` 已加入 `.gitignore`。設定檔內的模型路徑以設定檔所在目錄為基準，因此複製到其他資料夾時需一併調整相對路徑。
