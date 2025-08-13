# Traffic-Predict — Forecasting with ARIMA, LSTM, and Chronos-Bolt

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mehdislik/Traffic-Predict/blob/main/traffic-prediction.ipynb)

Predict the **last weekly traffic measurement per sector antenna** and compare models by:
- **Accuracy:** RMSE
- **Speed:** **inference time only** (training excluded for fairness with pretrained models)
- **Compute:** FLOPs proxies and a coarse CPU utilization proxy

The notebook is Colab-ready and runs end-to-end: data cleaning → ARIMA → LSTM → Chronos-Bolt (Tiny/Mini/Small/Base).

---

## Quick start

### Run on Google Colab
1. Click the **Open in Colab** badge above.
2. Upload `histo_trafic.csv` (and any other data files) to the Colab working directory.
3. Run all cells.

### Run locally
```bash
# Create env (optional)
python -m venv .venv && source .venv/bin/activate

# Install requirements
pip install -q pandas numpy scikit-learn statsmodels tensorflow torch psutil dateparser
pip install -q git+https://github.com/amazon-science/chronos-forecasting.git
````

Then open `traffic-prediction.ipynb` in Jupyter/VS Code and run the cells.

---

## Data

* **Input**: `histo_trafic.csv` with columns: `secteur, site, tstamp, trafic_mbps`
* **Cleaning** (notebook cell):

  * Normalizes column names
  * Parses French timestamps with `dateparser`
  * Converts `trafic_mbps` to numeric (comma → dot)
  * Drops invalid rows, removes duplicates and negative traffic
* **Output**: `histo_trafic_cleaned.csv`

>  Large or private datasets should not be committed to Git. Keep them local or use Git LFS.

---

## Models

* **ARIMA(2,1,2)** — classic statistical baseline (fast, linear)
* **LSTM (50 units)** — single-layer + Dense(1)
* **Chronos-Bolt** (Amazon):

  * Variants: `tiny`, `mini`, `small`, `base`
  * Inference-only, pretrained, uses `BaseChronosPipeline`
  * `p50` (median) used as the point forecast

---

## Evaluation metrics

* **RMSE** — primary accuracy metric
* **Inference time only** — measured with `time.perf_counter()`
* **Compute proxies**

  * **ARIMA**: \~`O(n*(p^2 + q^2))`
  * **LSTM**: \~`4 * (h^2 + h*x + h) * timesteps`
  * **Chronos-Bolt**: \~`2 * |θ| * L` (parameters × context length)
* **Power proxy**: average CPU utilization via `psutil` (illustrative only)

---

## Outputs (saved by the notebook)

* `histo_trafic_cleaned.csv`
* `ARIMA_complete_results.csv`
* `LSTM_complete_results.csv`
* `Chronos_models_comparison.csv`
* `Chronos_amazon_chronos-bolt-<variant>_complete_results.csv` (per variant)

Each CSV contains per-sector predictions, RMSE, inference time, and compute proxies.
The comparison CSV summarizes RMSE and inference latency across Chronos variants.

---

## Repo structure (suggested)

```
.
├─ traffic-prediction.ipynb        # Main Colab notebook
├─ data/                           # input data time series
├─ results/                        # CSV outputs
├─ README.md
└─ .gitignore
```

**.gitignore (suggested):**

```
.ipynb_checkpoints/
__pycache__/
*.pyc
data/
results/*.png
results/*.pdf
```

---

## Reproducing results

1. Run **data cleaning** cell → produces `histo_trafic_cleaned.csv`
2. Run **ARIMA** cell → saves `ARIMA_complete_results.csv`
3. Run **LSTM** cell → saves `LSTM_complete_results.csv`
4. Run **Chronos-Bolt** cell → saves per-variant CSVs + `Chronos_models_comparison.csv`

---

## Troubleshooting

* **CUDA not found / runs on CPU**
  In Colab: `Runtime → Change runtime type → GPU`. Chronos will use GPU if available.

* **Chronos install issues**
  Ensure: `pip install git+https://github.com/amazon-science/chronos-forecasting.git`.

---

## Notes & credits

* Chronos / Chronos-Bolt by **Amazon Science**
* This repo evaluates ARIMA, LSTM, and Chronos-Bolt on the same data and reports **inference-only** latency for fairness with pretrained models.


```
