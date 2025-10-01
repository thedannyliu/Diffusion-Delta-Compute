## Delta-Compute dLLM PoC 操作指南（最新版）

本文件提供完整的復刻流程，涵蓋環境設定、程式碼結構說明、各階段腳本指令、輸出檢查、視覺化產物，以及常見疑難排解。閱讀完畢後即可依章節循序完成 P0–P6 的實驗與報告。所有內容皆以本倉庫目前狀態（`src/run.py` 與 `docs/master_plan_poc_0929.md`）為準。

---

### 0. 五分鐘快速開始
1. **安裝環境**（詳見 §1）：
   ```bash
   conda create -y -n dcllm python=3.10
   conda activate dcllm
   pip install -r requirements.txt
   ```
2. **設定路徑**：複製或編輯 `configs/paths.yaml`，指向本機資料集、模型與輸出資料夾。
3. **下載資料/模型（可選）**：
   ```bash
   scripts/download_data_and_model.sh   # 需事先登入 HuggingFace/W&B（如有）
   ```
4. **執行 Smoke 測試**：
   ```bash
   scripts/run_p1_teacher.sh            # 產出教師軌跡與視覺化
   scripts/run_p2_rule_gate.sh          # 規則式 Gate
   scripts/run_p3_learned_gate.sh       # 訓練 + 套用學習式 Gate
   scripts/run_p4_adaptive.sh           # LTE 控制步數跳算
   scripts/run_p5_oracle.sh             # Oracle / Baseline 頭尾界線
   scripts/run_p6_combine.sh            # 匯整報表與 Pareto
   ```
5. **查看輸出**：`reports/<phase>/<mode>/<engine>/<run_id>/` 下含 JSONL、CSV 與圖檔。若有設定 W&B，儀表板會自動連動。

---

### 1. 環境與必要套件
- **Python**：3.10（建議使用 Conda 管理環境）。
- **GPU**：若需加速，請安裝 CUDA 12.x 對應之 PyTorch（>= 2.2）。
- **安裝步驟**：
  ```bash
  conda create -y -n dcllm python=3.10
  conda activate dcllm
  pip install -r requirements.txt
  ```
- **可選套件**：
  - `wandb`：自動同步指標與圖像。
  - `pynvml`：NVML 監測 GPU 使用率。
  - `torchvision` / `matplotlib`：完整視覺化支援。
- **路徑設定 (`configs/paths.yaml`)**：
  ```yaml
  data_root: /path/to/data
  models_root: /path/to/models
  outputs_root: /path/to/reports
  ```
  所有腳本會優先讀取此檔，若未提供則預設寫入 `reports/`。

---

### 2. 倉庫結構速覽
| 路徑 | 說明 |
| --- | --- |
| `src/run.py` | 單一入口點，支援 `teacher` / `rule_gate` / `learned_gate` / `adaptive` / `oracle` / `baselines` 模式。內建 Conformal 風險控制、預算控制與 LTE 排程。 |
| `src/gating/` | Gate 元件：規則式 (`rule_gate.py`)、學習式 (`learned_gate.py`)、Conformal 校正 (`conformal.py`)、預算控制 (`budget.py`)、回滾 (`rollback.py`)、自適應排程 (`adaptive.py`)。 |
| `src/engine/` | 引擎抽象層與實作：`mock_engine.py`（快速測試）、`d2f_engine.py`（整合 diffusion LLM）。 |
| `src/probes/` | 量測函式（cosine、ΔL2、KL、entropy、margin）。 |
| `src/eval/` | 資料集載入器與評分器（Wikitext‑2、LAMBADA-open、Tiny GSM8K 等）。 |
| `src/viz/` | 視覺化與表格輸出（Pareto、ROC/PR、Latency tails 等）。 |
| `configs/` | 任務設定 (`tasks.yaml`)、風險門檻 (`thresholds.yaml`)、路徑設定 (`paths.yaml`) 與各階段 smoke 組態（`p*_smoke.yaml`）。 |
| `scripts/` | 封裝命令（`run_p*_*.sh`）、Slurm 模板、環境腳本。 |
| `reports/` | 預設輸出目錄，依階段與模式分層存放 JSONL/CSV/圖檔。 |
| `docs/` | 專案文件（Master Plan、PoC 指南）。 |

---

### 3. 組態檔與關鍵參數
- **任務切換 (`configs/tasks.yaml`)**：定義資料集、樣本數、最大生成長度等。腳本可透過 `TASKS_CFG` 指向自訂 YAML。
- **門檻設定 (`configs/thresholds.yaml`)**：提供各資料集（wikitext2、lambada、gsm8k）的保守/平衡/積極檔位，腳本以 `PROFILE` 與 `DATASET_KEY` 選擇。
- **風險控制**：
  - `--risk_delta`：Conformal 容忍度 δ，預設 0.01，可調整至 0.005（Wikitext）等。
  - `--budget_fraction`：每步允許重新計算的 token 比例，預設 1.0（不限制）。
- **Adaptive 參數**：`--lte_eps`, `--lte_min_consec`, `--max_stride`, `--adaptive_budget`。

---

### 4. 各階段復刻流程
以下依 Master Plan（P0–P6）說明目的、腳本、主要輸出以及指標檢查。所有腳本皆可覆寫環境變數以便於批次測試。

#### P0 — 環境驗證與 Mock Smoke
- **目的**：確認依賴、Mock 引擎、報表/圖表流程正常。
- **步驟**：
  ```bash
  scripts/run_p1_teacher.sh ENGINE=mock OUT_DIR=reports/mock_smoke
  ```
- **輸出重點**：`teacher_summary_*.json`（延遲、吞吐）、`teacher_cosine_heatmap_*.png`。
- **檢查**：cosine/ΔL2 曲線隨步數趨於穩定；產生的 feature 檔案可供接續階段使用。

#### P1 — 教師軌跡與視覺化
- **腳本**：`scripts/run_p1_teacher.sh`。
- **常用變數**：
  - `ENGINE=d2f`（啟用真實模型）。
  - `TASKS_CFG=configs/tasks.yaml`（可指向子樣本設定）。
  - `ORACLE_EVAL=true`（同時產生 Oracle 頭界線特徵）。
- **產物**：
  - `reports/.../runs/teacher_*.jsonl`：逐步統計。
  - `reports/.../figures/`：熱圖、直方圖、LTE 殘差圖。
  - `reports/.../features/teacher_features_*.npz`：供 P3 訓練使用（含 σₜ 正規化特徵與風險分數）。
- **視覺化**：見 §6。建議同步至 W&B（自動完成）以利比較。

#### P2 — 規則式 Gate（Rule Gate）
- **腳本**：`scripts/run_p2_rule_gate.sh`
- **重要參數**：
  - `PROFILE=balanced DATASET_KEY=wikitext2` 對應 `configs/thresholds.yaml`。
  - `RISK_DELTA=0.006`，`BUDGET_FRACTION=0.6`。
  - `WATCHDOG_MIN_COS`、`WATCHDOG_MAX_KL` 覆寫守門人條件。
- **產物**：
  - `rule_gate_summary_*.json`：延遲、跳算比例、Conformal δ 觀測。
  - `rule_gate_savings_hist_*.png`：FLOPs 節省直方圖。
- **檢查**：`delta_violation_mean` 是否 ≤ 指定 δ；`skip_ratio_mean` 與 Teacher 比較品質差異（可用 `run_eval.sh` 補充指標）。

#### P3 — 輕量學習式 Gate
- **腳本**：`scripts/run_p3_learned_gate.sh`
  1. 讀取最新或指定的 `teacher_features_*.npz`。
  2. 執行 `src.learned.train_gate` 訓練（可調整 `TRAIN_EPOCHS` 等環境變數）。
  3. 套用 `src.run --mode learned_gate` 產生報表。
- **建議流程**：
  ```bash
  FEATURE_SOURCE="reports/P1/**/teacher_features_*.npz"   TRAIN_EPOCHS=12 TRAIN_LR=2e-3 TRAIN_THRESHOLD_GRID="0.5,0.6,0.7"   scripts/run_p3_learned_gate.sh
  ```
- **檢查**：
  - `learned_gate_summary_*.json` 中 `risk_threshold_mean` 與 δ 是否符合預期。
  - 機率直方圖確認預測分佈未崩壞。

#### P4 — 自適應步數排程（Adaptive Scheduling）
- **腳本**：`scripts/run_p4_adaptive.sh`
- **參數**：
  - `LTE_EPS=0.015`、`LTE_MIN_CONSEC=2`、`MAX_STRIDE=4`。
  - `ADAPTIVE_BUDGET=0.20`（總跳算步數上限 20%）。
- **輸出**：`adaptive_summary_*.json` 提供估計步數節省、延遲與設定。
- **檢查**：確認 `skip_ratio_est_mean` 與 δ 觀測（若超過預算需調整 `lte_eps` 或 `max_stride`）。

#### P5 — Oracle & Baseline（頭尾界線）
- **腳本**：
  - `scripts/run_p5_oracle.sh`：呼叫 `src.run --mode oracle`，以教師軌跡產生 Oracle Skip/Freeze 與 Pareto 上限。
  - `scripts/run_p5_baselines.sh`：產生固定步長、index-only、均勻層重算等參考曲線。
- **產物**：`oracle_summary_*.json`、`baselines_*.json`，內容含 `%FLOPs` 頭尾界線、品質比較線上使用。
- **檢查**：確保 Oracle 頭界線優於實際 Gate，且 Baseline 成效為下界。

#### P6 — Combine & Evaluate（總結與報告）
- **腳本**：`scripts/run_p6_combine.sh`
  - 讀取 Teacher/P2/P3/P4/Oracle/Baseline 的 JSON/CSV。
  - 產出綜合表格、Pareto、Compute-AUC、Quality@Budget 與失敗案例摘要。
- **輸出**：`reports/P6/combine/...` 下的 `tables/*.csv`、`figures/*.png`、`runs/final_metrics.json`。
- **檢查**：
  - `compute_auc`、`skip_regret` 是否優於 Baseline。
  - 三大任務（語言建模、遮蔽填字、推理）品質皆符合 KPI。

---

### 5. CLI 參數與腳本對應
以下列出常用環境變數／旗標，所有腳本皆支援額外的 CLI 透傳（見末段 `"$@"`）：

| 參數 | 說明 | 範例 |
| --- | --- | --- |
| `ENGINE` | `mock` 或 `d2f` | `ENGINE=d2f scripts/run_p2_rule_gate.sh` |
| `OUT_DIR` | 覆寫輸出根目錄 | `OUT_DIR=/data/runs scripts/run_p1_teacher.sh` |
| `TASKS_CFG` | 指定任務 YAML | `TASKS_CFG=configs/tasks_full.yaml` |
| `THRESHOLDS_CFG` | 指定門檻 YAML | `THRESHOLDS_CFG=configs/thresholds.yaml` |
| `PROFILE` / `DATASET_KEY` | 選擇門檻檔位與資料集 | `PROFILE=aggressive DATASET_KEY=wikitext2` |
| `RISK_DELTA` | Conformal δ | `RISK_DELTA=0.005 scripts/run_p2_rule_gate.sh` |
| `BUDGET_FRACTION` | 每步預算 | `BUDGET_FRACTION=0.6` |
| `LTE_EPS` / `LTE_MIN_CONSEC` / `MAX_STRIDE` | Adaptive 參數 | `LTE_EPS=0.012 MAX_STRIDE=3` |
| `FEATURE_SOURCE` | P3 訓練資料來源 | `FEATURE_SOURCE="reports/P1/**/teacher_features_*.npz"` |
| `WEIGHTS_OUT` | P3 權重輸出路徑 | `WEIGHTS_OUT=models/gates/wikitext_balanced.npz` |

---

### 6. 視覺化指南
1. **自動產物**：每次執行會在 `reports/.../figures/` 產生：
   - `teacher_cosine_heatmap_*.png`、`teacher_dl2_heatmap_*.png`：Layer × Step 平均值熱圖。
   - `*_savings_hist_*.png`：節省比例直方圖。
   - `*_pareto_*.png`：延遲 vs 品質 Pareto 曲線（P6）。
   - `roc_pr_*.png`、`calibration_curve_*.png`：P3 學習式 Gate 訓練產物。
   - `lte_residual_heatmap_*.png`：Adaptive 模式的 LTE 觀測。
2. **表格資料**：`reports/.../runs/*.csv` 內含原始統計，可用 `pandas` 或 `src/viz/tables.py` 生成客製圖表。
3. **W&B 儀表板**（可選）：若環境已登入 `wandb`，`src/run.py` 會自動建立 run，指標包含 latency、skip_ratio、risk_delta 等。建議建立儀表板以叢列比較 Teacher / Rule / Learned / Adaptive 模式。
4. **Compute-AUC 與 Quality@Budget**：P6 腳本會在 `reports/P6/.../tables/` 內輸出 CSV，含各模式於不同 FLOPs 比率下的品質。匯入至 Google Sheets 或 Notebook 後繪製折線，可快速對照 Oracle 頭界線。
5. **失敗案例（Failure Gallery）**：`src/viz/failure_gallery.py` 會挑選品質或一致性下降的樣本，輸出差異摘要（包含注意力權重）。如需人工檢視，請打開對應 JSON/PNG。

---

### 7. 常見問題與排解
| 問題 | 排解 |
| --- | --- |
| `ModuleNotFoundError: yaml` | 執行 `pip install pyyaml` 後重試。 |
| `Learned gate` 訓練找不到 features | 確認已執行 P1，或手動指定 `FEATURE_SOURCE`。 |
| δ 超過目標 | 降低 `RISK_DELTA`、或使用更保守 `PROFILE`；檢查 `rule_gate_summary` 中的 `delta_violation_mean`。 |
| Adaptive 無效 | 測試時若 `max_stride=1` 或 `lte_eps` 設太緊會導致無跳算；可先用 mock 引擎調整。 |
| 圖表未生成 | 安裝 `matplotlib` 及 `seaborn`，或使用 Notebook 透過 CSV 自行繪製。 |
| GPU 利用率為 0 | 確認 `pynvml` 已安裝並啟用，或以 `nvidia-smi dmon` 方式記錄。 |

---

### 8. 推薦工作流程
1. **每日 Smoke**：以 mock 引擎驗證腳本／視覺化是否壞掉。
2. **Teacher（P1）**：確保資料集、模型與輸出體系正常，並產出最新 features。
3. **Rule Gate（P2）**：根據 `rule_gate_summary` 微調 τ/ρ/m/K；若 δ 超標就回去調整門檻或 watchdog。
4. **Learned Gate（P3）**：確認訓練 ROC/PR 與校準曲線、儲存最佳權重。
5. **Adaptive（P4）**：搭配風險分數調整 stride，觀察 LTE 分布。
6. **Oracle/Baseline（P5）**：提供報告所需的上/下界參考線，確保 gating 優於 baseline。
7. **Combine（P6）**：整理 CSV/圖表，產出生產級報表與 Pareto；最後可使用 `docs/master_plan_poc_0929.md` 檢查 KPI 是否達成。

---

### 9. 深入閱讀
- `docs/master_plan_poc_0929.md`：完整研究里程碑、KPI 與實驗設計。
- `src/gating/conformal.py`、`src/gating/budget.py`：Conformal 風險控制與預算分配實作。
- `src/viz/plots.py`：Plot API，可客製更多圖表。
- `scripts/run_eval.sh`：若需額外評測，可參考此腳本整合多任務指標。

---

如需更多協助，建議建立 Notebook（載入 `reports/...` CSV）進行互動式探索，或整合自動化 pipeline（CI）定期執行 P0–P3 Smoke 測試。
