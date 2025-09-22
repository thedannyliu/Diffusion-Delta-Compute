## Delta-Compute dLLM PoC 指南（繁體中文）

本指南說明如何在本倉庫完成從 P0 到 P3 的原型實作與評測，包含資料集載入、規則式凍結（Rule Gate）、輕量學習式 Gate，以及必要的可視化與效能量測。

### 0. 環境準備
- Python 3.10（建議 Conda）
- CUDA 12.x + PyTorch >= 2.2
- 安裝依賴：
```bash
conda create -y -n dcllm python=3.10
conda activate dcllm
pip install -r requirements.txt
```

注意：若要使用 GPU 版 torch，請依機器環境替換 torch 版本／索引來源（例如 PyTorch 官方 CUDA whl）。

### 1. 倉庫結構（重點）
- `src/engine/`: 引擎抽象與實作（`base_engine.py`, `mock_engine.py`, `d2f_engine.py`）。
- `src/probes/`: 量測指標（cosine、ΔL2、KL、entropy、margin）。
- `src/gating/`: Gate 模組（P2 規則式）。
- `src/eval/`: 資料集載入與評分器（Wikitext‑2、LAMBADA、GSM8K）。
- `src/viz/`: 圖表輸出（無 matplotlib 時降級為 .npy/.csv）。
- `src/run.py`: 入口點，支援 `teacher` 與 `rule_gate` 模式。
- `configs/`: 門檻與任務設定樣例。
- `scripts/`: 快速腳本與 Slurm 模版。

### 2. P1 — Teacher 路徑量測與可視化
目的：在不跳算的情況下，記錄各步之間（step t 與 t-1）的表徵變化，用以理解「穩定度」與後續 Gate 的依據。

指令（小型 Smoke 測試）：
```bash
python -m src.run --mode teacher --engine mock --out_dir reports \
  --num_prompts 16 --max_new_tokens 128 --num_steps 12
```

輸出：
- `reports/<run>/runs/teacher_*.jsonl`：每步的 cosine/ΔL2/KL/entropy/margin 平均值。
- `reports/<run>/figures/*`：熱圖或直方圖（若無 matplotlib，輸出 .npy/.csv）。

檢查：
- Cosine 遞增（越接近 1）或 ΔL2 遞減的趨勢。
- KL 與 entropy 隨步數下降的趨勢。

內部一致性指標：
- 在 `teacher` 模式下，`Final tokens 一致率` 應為 100%（與自身基準一致）。
- `Step-wise KL / entropy` 的曲線要合理（避免量測實作錯誤）。

### 3. P2 — 規則式凍結（Rule-based Gating）
規則：若 `(cos ≥ τ) 或 (ΔL2 ≤ ρ)` 且連續滿足 `m` 步，則凍結該 token `K` 步；每步更新看守（watchdog），若 `(cos < 0.90) 或 (KL > 0.02)`，立即解除凍結並重算。

指令：
```bash
python -m src.run --mode rule_gate --engine mock --out_dir reports \
  --num_prompts 16 --max_new_tokens 128 --num_steps 12 \
  --tau 0.95 --rho 0.10 --m 2 --freeze_K 2 \
  --watchdog_min_cos 0.90 --watchdog_max_kl 0.02
```

輸出：
- `savings` 直方圖：顯示已跳過（token×step）比例。
- 後續可擴充：記錄 per-layer 跳過次數，形成（token×layer×step）比例。

效能數據（建議記錄）：
- `Wall-clock latency`（ms/句子，batch=1）。
- `Throughput`（seq/s，batch=4–8）。
- `已跳過比例`（token×layer×step）（P2 後持續記錄）。
- `GPU 利用率`（以 `nvidia-smi` 輕量查詢）。

### 4. 真實引擎：d2f-dream 整合
`src/engine/d2f_engine.py` 提供 D2F 封裝骨架，需將其中 `TODO` 替換為實際 d2f-dream API 呼叫：
1) 初始化：載入 d2f 模型與裝置。
2) `encode_prompt`：將文字轉為引擎 state。
3) `step(state, t, compute_mask)`: 以引擎每步 API 前進一個 diffusion step，並支援 `compute_mask` 的 per-token 凍結。
4) `decode`：將最終 state 轉回文字（必要時）。

完成後即可用相同指令替換 `--engine d2f` 跑真實模型。PoC 初期建議先以 batch=1 驗證時間，穩定後再測 throughput。

### 5. P3 — 輕量學習式 Gate（Lightweight Learned Gate）
資料蒐集（離線）：
- 用 `teacher` 路徑記錄每步特徵：`cos, ΔL2, KL, entropy, margin`，可選 `attn` 統計。
- 標註：若在 (t,i) 凍結未影響最終任務成功（或 logits/最終分數無變動），標為 `safe_to_freeze=1`。

模型：
- 邏輯回歸或兩層 MLP（隱層約 64，總參數 <50k），輸出 `p_freeze` 與（可選）凍結長度 {1,2,3}。

部署（線上）：
- 以批次特徵丟入 gate，得到每 token 的 `p_freeze` 與長度，更新 `compute_mask` 供引擎在下一步使用。
- 維持 watchdog 機制以避免品質劣化。

預期成果：
- 三類任務（Wikitext‑2 PPL、LAMBADA acc、Tiny GSM8K EM）上的 `質-速` 曲線：在 ≥1.5× 加速下，品質衰減在容忍範圍（例如 PPL +≤3%、acc/EM 降幅 ≤1%）。
- 可視化圖：
  - Cosine/ΔL2 熱圖、直方圖。
  - Token 穩定步長分佈（連續滿足條件的長度）。
  - FLOPs/跳算比例 vs 品質曲線；Latency vs 品質的 Pareto 前沿。

### 6. 資料集載入與評測
已提供：
- `src/eval/tasks.py`: `load_wikitext2`, `load_lambada_openai`, `load_gsm8k_tiny`。
- `src/eval/scorer.py`: 困惑度計算（由 NLL 推得）、EM/acc 計算。

PPL：
- 以模型對文本計算平均負對數似然（NLL），`ppl = exp(mean_nll)`。

LAMBADA：
- 將句子最後一詞作為 label，模型需預測最後一詞是否命中；以 acc 評估。

GSM8K（Tiny）：
- 以簡化規則解析最終答案（'####' 後的數字）；採用 EM 作為指標。

### 7. 實驗建議與檔案位置
- 每次實驗請固定 seed 與保存 `configs/*.yaml`。
- 所有輸出（表格、圖）集中放置於 `reports/<run>/`。
- 重要指標請整理為 CSV，方便後續合併成報告與 Pareto 圖。

### 8. 風險與守則
- 避免過度凍結導致錯誤累積：啟用 watchdog 與週期性重算（K/M）。
- 對注意力中心度高的 token 可降低凍結概率（可在 learned gate 納入 attn 特徵）。
- 保持 Gate 計算開銷 < 3% 總延遲（向量化實作，避免 Python per-token 迴圈）。


