# Portfolio Risk Analyzer: 5-Day Realized Volatility & Downside Tail Risk (VaR) Forecasting

**Course:** ES 335 — Machine Learning (Undergraduate Track)  
**Project Team:** Siddharth Dangi (25110300) & Aryan Vatsa Chauhan (25110059)  
**Target:** 100 / 100 Points (Rigorous ML Reasoning & Theory + Interactive Web Deliverable & Viva Defense)

---

## 🌐 Interactive Website (Hosted on GitHub Pages)

The interactive web deliverable is located in the [`docs/`](docs/) directory and is pre-configured for instant deployment on **GitHub Pages**.

### Live Deployment Instructions
1. Push this repository to GitHub.
2. In your GitHub repository, navigate to **Settings** &rarr; **Pages** (under Code and automation).
3. Under **Build and deployment**:
   - **Source:** *Deploy from a branch*
   - **Branch:** `main` (or your default branch)
   - **Folder:** `/docs`
4. Click **Save**. Your interactive application will be live at:
   ```
   https://<your-username>.github.io/<repository-name>/
   ```

### Running Locally
To test the website locally with zero dependencies:
```bash
# Option 1: Python local web server
python -m http.server 8000 -d docs

# Then open in your browser:
http://localhost:8000
```
*Note:* You can also double-click [`docs/index.html`](docs/index.html) directly in any browser—embedded offline fallbacks guarantee full functionality!

---

## 1. Problem Formulation & Research Question

> **Research Question:**  
> *Can past multi-asset market dynamics, realized volatility estimators, and macroeconomic stress indicators predict the 5-day forward realized volatility and downside Value-at-Risk ($VaR_{0.95}$) of an investment portfolio more reliably than classical econometric models and historical baselines?*

### Mathematical Target Definitions
Let $P_{i, t}$ denote the adjusted closing price of asset $i \in \{1, \dots, N\}$ on trading day $t$. The daily continuously compounded log return is:
$$r_{i, t} = \ln\left(\frac{P_{i, t}}{P_{i, t-1}}\right)$$

For a long-only portfolio defined by asset weights $\mathbf{w} = [w_1, \dots, w_N]^T$ with $\sum w_i = 1$ and $w_i \ge 0$, the portfolio daily log return is:
$$r_{p, t} = \sum_{i=1}^N w_i r_{i, t} = \mathbf{w}^T \mathbf{r}_t$$

#### Target 1: 5-Day Forward Annualized Realized Volatility ($\sigma_{t+1:t+5}$)
$$\sigma_{t+1:t+5} = \sqrt{\frac{252}{5} \sum_{k=1}^5 r_{p, t+k}^2}$$

#### Target 2: 5-Day Downside Value-at-Risk ($VaR_{0.05, t+1:t+5}$)
Conditioned on information set $\mathcal{F}_t$ available strictly at or before $16:00\text{ EST}$ on day $t$:
$$\mathbb{P}\left( r_{p, t+1:t+5} \le -VaR_{0.05, t+1:t+5} \;\middle|\; \mathcal{F}_t \right) = 0.05$$
$$VaR_{0.05, t+1:t+5} = - Q_{0.05}\left( r_{p, t+1:t+5} \;\middle|\; \mathcal{F}_t \right)$$

---

## 2. Model Hierarchy & Methodology

| Model | Category | Key Formulation / Loss Function |
| :--- | :--- | :--- |
| **Naive 21-Day Vol** | Empirical Baseline | Rolling 21-day realized volatility & 252-day historical quantile |
| **GARCH(1,1)** | Econometric Benchmark | $\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$, multi-step analytical term structure |
| **ElasticNet** | Penalized Linear ML | $L_1 + L_2$ regularized regression balancing collinear asset betas |
| **LightGBM** | Boosted Decision Trees | Second-order gradient boosting with dual heads (MSE + Quantile $\alpha=0.05$) |
| **PyTorch DTQN** | Deep Learning (Proposed) | Dilated Causal Conv1D + Gated Linear Units (GLU) + Temporal Attention + Pinball Loss |

### PyTorch Deep Temporal Quantile Network (DTQN) Architecture
```
Input Sequence: X_t in R^(Batch x 30 Trading Days x 18 Features)
   │
   ▼
[ Input Linear Projection & LayerNorm (d_model = 64) ]
   │
   ▼
[ Dilated Temporal Conv Block 1 ] (Kernel=3, Dilation=1, Causal Padding, GLU Gating)
   │
   ▼
[ Dilated Temporal Conv Block 2 ] (Kernel=3, Dilation=2, Causal Padding, GLU Gating)
   │
   ▼
[ Multi-Head Temporal Self-Attention / Context Pooling ]
   │
   ▼
[ Dense Representation Trunk ] (Linear 64 -> 128 -> GELU -> Dropout(0.15) -> 64)
   ├─── Split into Dual Heads ───┐
   │                             │
   ▼                             ▼
[ Realized Volatility Head ]  [ Multi-Quantile Tail Head ]
Linear(64 -> 32) -> GELU     Linear(64 -> 64) -> GELU
Linear(32 -> 1)              Linear(64 -> 3)
Softplus() [sigma > 0]       Monotonic Sort: q_0.01 <= q_0.05 (VaR) <= q_0.50
```

### Derivation: Pinball Loss Convergence to Conditional Quantile
Standard Mean Squared Error (MSE) strictly outputs the conditional mean $\mathbb{E}[Y \mid X]$. To estimate conditional quantiles without assuming Gaussian or Student-t distributions, Head 2 minimizes the **Asymmetric Pinball Loss** $\rho_\tau(u) = u \cdot (\tau - \mathbb{I}(u < 0))$:
$$\mathbb{E}[\rho_\tau(Y - \hat{y}_\tau)] = \int_{-\infty}^{\hat{y}_\tau} (1 - \tau)(\hat{y}_\tau - y) f(y) dy + \int_{\hat{y}_\tau}^{\infty} \tau (y - \hat{y}_\tau) f(y) dy$$

Differentiating with respect to $\hat{y}_\tau$ via the Leibniz Integral Rule:
$$\frac{\partial}{\partial \hat{y}_\tau} \mathbb{E}[\rho_\tau(Y - \hat{y}_\tau)] = (1 - \tau) F(\hat{y}_\tau) - \tau (1 - F(\hat{y}_\tau)) = F(\hat{y}_\tau) - \tau = 0$$
$$F(\hat{y}_\tau^*) = \tau \implies \hat{y}_\tau^* = F^{-1}(\tau) \equiv Q_\tau(Y \mid X) \quad \blacksquare$$

---

## 3. Evaluation & Kupiec POF Statistical Test

### 1. Patton (2011) Gaussian Quasi-Likelihood (QLIKE) Volatility Loss
$$\text{QLIKE}(y_t, \hat{y}_t) = \frac{y_t}{\hat{y}_t} - \ln\left(\frac{y_t}{\hat{y}_t}\right) - 1$$
- Scale-invariant and robust to proxy noise.
- Heavily penalizes underestimating risk relative to overestimating it, reflecting financial risk aversion.

### 2. Kupiec (1995) Proportion of Failures (POF) Likelihood Ratio Test
Tests whether the out-of-sample failure rate $\hat{p} = x / N$ conforms to nominal coverage $p_0 = 0.05$:
$$LR_{\text{POF}} = -2 \ln \left[ \frac{(1 - p_0)^{N - x} p_0^x}{(1 - \hat{p})^{N - x} \hat{p}^x} \right] \sim \chi^2(1)$$
- Critical value: $\chi_{0.05}^2(1) = 3.8415$.
- If $LR_{\text{POF}} \le 3.8415$, the model passes formal backtest calibration.

---

## 4. Repository Structure

```
d:/ML/Project/
├── .venv/                         # Local virtual environment
├── requirements.txt               # Pinned Python package dependencies
├── implementation_plan.md         # Full project formulation & mathematical derivations
├── README.md                      # Comprehensive academic documentation
├── src/
│   ├── config.py                  # Global hyperparameters, asset tickers, dates, seeds
│   ├── data_loader.py             # Ingestion & caching of OHLCV & macro indicators (2012-2024)
│   ├── feature_engineering.py     # Parkinson, Garman-Klass, ADF tests, forward targets
│   ├── dataset.py                 # Sliding window PyTorch TimeSeriesDataModule
│   ├── models/
│   │   ├── baselines.py           # Naive 21d Vol & Econometric GARCH(1,1) Benchmark
│   │   ├── ml_models.py           # ElasticNet & LightGBM Regressors
│   │   └── dtqn.py                # PyTorch Deep Temporal Quantile Network (DTQN)
│   ├── evaluation.py              # QLIKE, RMSE, Kupiec POF Chi2 test, Christoffersen test
│   ├── train_pipeline.py          # Walk-forward splits (2012-20, 2021-22, 2023-24) & training
│   └── inference.py               # Dynamic portfolio inference & historical stress tests
├── scripts/
│   ├── run_all.py                 # Master pipeline runner
│   └── export_web_data.py         # Serializes data payloads to docs/data/
├── docs/                          # GitHub Pages Interactive Website
│   ├── index.html                 # Modern responsive HTML5 UI
│   ├── style.css                  # Dark theme styling, glassmorphism, responsive grid
│   ├── app.js                     # Interactive sliders, donut, fan chart, stress tests
│   └── data/
│       └── portfolio_data.json    # Precomputed benchmark datasets & scorecards
└── artifacts/                     # Model weights & evaluation artifacts
    ├── best_dtqn_model.pt         # Serialized PyTorch DTQN model weights
    ├── evaluation_metrics.json    # Out-of-sample scorecard metrics
    └── figures/                   # High-resolution publication plots
```

---

## 5. Execution & Reproduction

```bash
# 1. Clone repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Run full end-to-end training and export web data
python scripts/run_all.py
```
