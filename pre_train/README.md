# 預訓練 policy

每個模型目錄都保留 `policy.pt` 與對應的 `metadata.json`。部署時會檢查模型 SHA256，以及觀測、關節順序、PD 增益、控制週期等合約；請將兩個檔案一起更新。

| 模型 | 狀態 | 檔案大小 | idle control |
|---|---|---:|---|
| `wtw/wtw_stand_still` | 預設 YAML 使用的模型 | 約 13.9 MiB | 未提供站立／踏步切換 |
| `wtw/wtw_idle_consistent_v3` | 候選模型，未自動啟用 | 約 10.6 MiB | 支援 `idle_step` 0／1 |

兩份模型都隨 repository 保存，合計約 24.5 MiB；下載後可直接執行 `python little_white_wtw.py --check`。不需要訓練專案或原始 checkpoint 來載入 TorchScript。

metadata 的 `source_checkpoint` 與 `sources` 保留匯出來源追溯資訊，其中可能是原訓練電腦的絕對路徑；runtime 不會依這些路徑讀檔。實際讀取的模型與 metadata 路徑由 YAML 指定，並以 YAML 所在目錄為基準解析。

## 模型 SHA256

```text
ef8f1e04210c3b3c6fdfb9f2478b1a52c105a1c5b81784f0a15609ebb7660051  wtw/wtw_stand_still/policy.pt
c35aae79d427a68baf8545c3dd9e9e36e277d9a478f105fc542b4fbe6da75c58  wtw/wtw_idle_consistent_v3/policy.pt
```

切換模型時，須同時調整 YAML 的 `policy_path`、`metadata_path` 與任何不同的合約設定，並先執行 `--check`。候選模型支援的 idle 功能由 metadata 偵測，不能在不支援的模型上直接打開。
