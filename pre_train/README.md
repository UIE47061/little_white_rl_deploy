# 部署模型

目前只保留兩個入口實際使用的模型。每顆模型的 `policy.pt` 與 `metadata.json` 需成對保存。

| 模型目錄 | 入口 | 輸入 | 狀態 |
|---|---|---|---|
| `mjlab/omni` | `little_white_mjlab.py` | `[batch,20,45]` | 短程 MuJoCo＋DDS 通路通過；訓練前處理仍待完整設定確認 |
| `wtw/wtw_slippery_trot_v8` | `little_white_wtw.py` | WTW 70 維 × 30 幀 | 滑地候選 seed88/update2000；部分左轉未通過整體驗收 |

MJLab 的 normalization 已內含；WTW 使用自身觀測格式。不能只替換 PT 而沿用另一版本的觀測或控制設定。
WTW v8 的 `acceptance.json` 與 `comparison.md` 保留來源評估結果；不是實機驗收證明。
來源 metadata 可能記錄原訓練電腦的絕對路徑，runtime 實際載入路徑由 YAML 指定。

## 檔案驗證

```text
808ac4a3fc1c5d0c24a2c7b7f40864c545ce0d7e7939b3857cb6e64383bf7284  mjlab/omni/policy.pt
2451b3405e79a7255f5996504973fb9194a7495e242528766dfa97bd404fbc59  wtw/wtw_slippery_trot_v8/policy.pt
```

```bash
python little_white_mjlab.py --check
python little_white_wtw.py --check
```

兩個入口在載入時檢查 SHA256 與模型介面。更換模型時需同步核對 YAML、metadata、關節順序與控制週期。
