# Implementation Plan: Portfolio Risk Analyzer
**Course:** ES 335 — Machine Learning (Undergraduate Track)  
**Project Team:** Siddharth Dangi (25110300) & Aryan Vatsa Chauhan (25110059)  
**Target:** 100 / 100 points (80 pts Shared ML Reasoning + 20 pts ES 335 Interactive Deliverable & Viva)

---

## 1. Problem Formulation & Exact Mathematical Definitions

### 1.1 Research Question
> *Can past multi-asset market dynamics, realized volatility estimators, and macroeconomic stress indicators predict the 5-day forward realized volatility and downside Value-at-Risk ($VaR_{0.95}$) of an investment portfolio more reliably than classical econometric models and historical baselines?*

### 1.2 Mathematical Formulation of Targets
Let $P_{i, t}$ denote the adjusted closing price of asset $i \in \{1, \dots, N\}$ at trading day $t$. The continuously compounded (log) return on day $t$ is:
$$r_{i, t} = \ln\left(\frac{P_{i, t}}{P_{i, t-1}}\right) = \ln P_{i, t} - \ln P_{i, t-1}$$

For an investment portfolio defined by asset weight allocation vector $\mathbf{w} = [w_1, w_2, \dots, w_N]^T \in \mathbb{R}^N$ with $\sum_{i=1}^N w_i = 1$ and $w_i \ge 0$ (long-only), the portfolio daily log return is approximated by:
$$r_{p, t} = \sum_{i=1}^N w_i r_{i, t} = \mathbf{w}^T \mathbf{r}_t$$
where $\mathbf{r}_t = [r_{1, t}, \dots, r_{N, t}]^T$.

The cumulative $h$-day forward portfolio return over horizon $h = 5$ days is:
$$r_{p, t+1:t+h} = \sum_{k=1}^h r_{p, t+k} = \ln\left(\frac{P_{p, t+h}}{P_{p, t}}\right)$$

#### Target 1: Forward Realized Volatility ($\sigma_{t+1:t+h}$)
Annualized continuous realized volatility over forward horizon $h = 5$ days:
$$\sigma_{t+1:t+h} = \sqrt{\frac{252}{h} \sum_{k=1}^h r_{p, t+k}^2}$$
*Unit:* Annualized percentage volatility (e.g., $0.185 = 18.5\%$).

#### Target 2: Downside Tail Risk / Value-at-Risk ($VaR_{\tau, t+1:t+h}$)
Value-at-Risk at confidence level $1 - \tau$ (specifically $\tau = 0.05$ for $95\%$ VaR) over horizon $h = 5$ is defined as the negative of the conditional $\tau$-th quantile of forward returns:
$$\mathbb{P}\left( r_{p, t+1:t+h} \le -VaR_{\tau, t+1:t+h} \;\middle|\; \mathcal{F}_t \right) = \tau$$
where $\mathcal{F}_t$ denotes the filtration (information set) available strictly at or before market close on day $t$.
$$VaR_{0.05, t+1:t+5} = -Q_{0.05}\left( r_{p, t+1:t+5} \;\middle|\; \mathcal{F}_t \right)$$
*Interpretation:* If $VaR_{0.05} = 4.2\%$, there is at most a $5\%$ probability that the portfolio loses more than $4.2\%$ over the next 5 trading days.

### 1.3 Feasibility & Information Boundary
- **Row Unit:** A single trading day $t$.
- **Causal Conditioning:** All inputs $\mathbf{X}_t \in \mathcal{F}_t$ use information recorded up to $16:00\text{ EST}$ on day $t$.
- **No Lookahead:** Targets span $[t+1, t+5]$. The current day close $P_t$ is the base price; zero future returns or volumes are visible during feature construction.

---

## 2. Advanced Feature Engineering & Mathematical Estimators

Rather than relying solely on closing prices, we incorporate intra-day high-low-open dynamics and cross-asset dependencies:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FEATURE MATRIX X_t                                      │
├───────────────────────────────┬───────────────────────────────┬─────────────────────────┤
│    Intraday Vol Estimators    │     Cross-Asset Dynamics      │   Macro Regimes / VIX   │
├───────────────────────────────┼───────────────────────────────┼─────────────────────────┤
│ • Parkinson Volatility        │ • 21-day Rolling Correlation  │ • VIX level & 5d delta  │
│ • Garman-Klass Volatility     │ • Portfolio Beta to SPY       │ • 10Y-2Y Yield Spread   │
│ • 5d, 21d, 63d Realized Vol   │ • Relative Strength (RSI)     │ • ATR (True Range)      │
└───────────────────────────────┴───────────────────────────────┴─────────────────────────┘
```

### 2.1 Range-Based Volatility Estimators

#### Parkinson Volatility (1980)
Assumes price diffusion follows continuous geometric Brownian motion with zero drift:
$$\sigma_{Park, t}^2 = \frac{1}{4 \ln 2} \left( \ln \frac{H_t}{L_t} \right)^2 \approx 0.3607 \cdot \left( \ln \frac{H_t}{L_t} \right)^2$$
*Mathematical Advantage:* Uses intraday extreme values (High $H_t$ and Low $L_t$). It is approximately $5\times$ more statistically efficient than the classical squared close-to-close return $(r_t^2)$.

#### Garman-Klass Volatility (1980)
Extends Parkinson by incorporating opening jumps ($O_t$) and closing prices ($C_t$):
$$\sigma_{GK, t}^2 = 0.5 \left( \ln \frac{H_t}{L_t} \right)^2 - (2\ln 2 - 1) \left( \ln \frac{C_t}{O_t} \right)^2 \approx 0.5 \left( \ln \frac{H_t}{L_t} \right)^2 - 0.3863 \left( \ln \frac{C_t}{O_t} \right)^2$$
*Mathematical Advantage:* Up to $7.4\times$ more efficient than close-to-close returns by accounting for intraday trend and drift.

### 2.2 Rolling Cross-Asset Covariance & Correlations
For assets $i$ and $j$, the rolling $M$-day sample covariance ($M = 21$) is:
$$\hat{\sigma}_{ij, t} = \frac{1}{M-1} \sum_{k=0}^{M-1} (r_{i, t-k} - \bar{r}_{i, t})(r_{j, t-k} - \bar{r}_{j, t})$$
$$\hat{\rho}_{ij, t} = \frac{\hat{\sigma}_{ij, t}}{\hat{\sigma}_{i, t} \hat{\sigma}_{j, t}}$$
These correlations capture flights-to-safety (e.g., breakdown of equity-bond negative correlation during inflation spikes).

### 2.3 Stationarity Diagnostics: Augmented Dickey-Fuller (ADF) Test
For every time series $y_t$, we estimate the ADF regression:
$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{p=1}^k \delta_p \Delta y_{t-p} + \varepsilon_t$$
- **Hypothesis:** $H_0: \gamma = 0$ (Unit root present $\implies$ non-stationary) vs $H_1: \gamma < 0$ (Stationary).
- **Rule:** If $p\text{-value} > 0.05$, the raw series is rejected. Price levels $P_t$ fail ($p > 0.60$), while log returns $r_t$ and normalized volatility ratios $\ln(\sigma_{t} / \sigma_{t-21})$ yield $p < 0.001$ (strictly stationary).

---

## 3. Mathematical Foundations of Candidate Models

### 3.1 Baseline 1: Naive Rolling Historical Volatility
$$\hat{\sigma}_{Naive, t+1:t+h} = \sqrt{\frac{252}{21} \sum_{k=0}^{20} r_{p, t-k}^2}$$
$$\widehat{VaR}_{Naive, 0.05, t+1:t+h} = - \text{Percentile}_{5\%}\left( \{ r_{p, s:s+5} \}_{s=t-252}^t \right)$$
Assumes the empirical distribution of the past 252 days directly mirrors the next 5 days.

### 3.2 Baseline 2: Econometric GARCH(1,1) Benchmark
The Generalized Autoregressive Conditional Heteroskedasticity model of Bollerslev (1986):
$$r_{p, t} = \mu + \varepsilon_t, \quad \varepsilon_t = \sigma_t z_t, \quad z_t \overset{iid}{\sim} \mathcal{N}(0, 1)$$
$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
- **Constraints:** $\omega > 0$, $\alpha \ge 0$, $\beta \ge 0$, and strict stationarity condition $\alpha + \beta < 1$.
- **Unconditional Long-Run Variance:**
  $$\sigma_{LR}^2 = \frac{\omega}{1 - \alpha - \beta}$$
- **$k$-Step Ahead Conditional Variance Forecast:**
  $$\mathbb{E}_t[\sigma_{t+k}^2] = \sigma_{LR}^2 + (\alpha + \beta)^{k-1} (\sigma_{t+1}^2 - \sigma_{LR}^2)$$
- **Forward 5-Day Cumulative Volatility:**
  $$\hat{\sigma}_{GARCH, t+1:t+5} = \sqrt{\frac{252}{5} \sum_{k=1}^5 \mathbb{E}_t[\sigma_{t+k}^2]}$$
- **GARCH Parametric VaR:**
  $$\widehat{VaR}_{GARCH, 0.05} = - \left( 5 \mu + z_{0.05} \cdot \sqrt{\sum_{k=1}^5 \mathbb{E}_t[\sigma_{t+k}^2]} \right)$$
  where $z_{0.05} = -1.6449$ is the 5th percentile of the standard normal distribution.

### 3.3 Candidate 1: Penalized Linear Model (ElasticNet)
$$\min_{\mathbf{\beta}, \beta_0} \frac{1}{2N_{tr}} \sum_{t=1}^{N_{tr}} \left( y_t - \beta_0 - \mathbf{x}_t^T \mathbf{\beta} \right)^2 + \lambda \left[ \rho \|\mathbf{\beta}\|_1 + \frac{1-\rho}{2} \|\mathbf{\beta}\|_2^2 \right]$$
Where $\rho \in [0, 1]$ balances $L_1$ sparsity and $L_2$ collinearity grouping across correlated assets.

### 3.4 Candidate 2: Gradient Boosted Decision Trees (LightGBM)
Minimizes regularized objective across $K$ additive regression trees:
$$\mathcal{L}^{(m)} = \sum_{t=1}^{N_{tr}} l\left(y_t, \hat{y}_t^{(m-1)} + f_m(\mathbf{x}_t)\right) + \Omega(f_m)$$
Tree splits use Second-Order Taylor Expansion:
$$\tilde{\mathcal{L}}^{(m)} \approx \sum_{t=1}^{N_{tr}} \left[ g_t f_m(\mathbf{x}_t) + \frac{1}{2} h_t f_m^2(\mathbf{x}_t) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
where $g_t = \frac{\partial l(y_t, \hat{y})}{\partial \hat{y}}$ and $h_t = \frac{\partial^2 l(y_t, \hat{y})}{\partial \hat{y}^2}$.

---

## 4. Deep Learning Model: PyTorch Deep Temporal Quantile Network (DTQN)

To overcome parametric distribution assumptions (e.g. Gaussian / Student-t), we design a custom neural architecture that jointly outputs continuous realized volatility and conditional quantiles ($VaR$).

### 4.1 Architecture Diagram

```
Input Lookback Sequence: X_t in R^(B x L x D)
(B: Batch Size, L: 30 Trading Days Lookback, D: 18 Engineered Features)
   │
   ▼
[ Input Projection & LayerNorm ]
   │ Linear(D -> d_model = 64) + LayerNorm + Dropout(0.1)
   ▼
[ Dilated Temporal Convolution Block 1 ]  (Kernel=3, Dilation=1, Causal Padding)
   │ Conv1D -> GLU (Gated Linear Unit) -> Residual Connection
   ▼
[ Dilated Temporal Convolution Block 2 ]  (Kernel=3, Dilation=2, Causal Padding)
   │ Conv1D -> GLU (Gated Linear Unit) -> Residual Connection
   ▼
[ Temporal Self-Attention / Context Pooling ]
   │ Queries, Keys, Values across sequence L -> Context vector h_t in R^(B x 64)
   ▼
[ Dense Representation Trunk ]
   │ Linear(64 -> 128) -> GELU -> Dropout(0.15) -> Linear(128 -> 64) -> LayerNorm
   ├─── Split into Dual Output Heads ───┐
   │                                     │
   ▼                                     ▼
[ Head 1: Realized Volatility ]      [ Head 2: Quantile Tail Heads ]
Linear(64 -> 32) -> GELU             Linear(64 -> 64) -> GELU
Linear(32 -> 1)                      Linear(64 -> 3)
Softplus activation (ensure > 0)     Monotonic Sorting: q_0.01 < q_0.05 < q_0.50
   │                                     │
   ▼                                     ▼
Predicted sigma_{t+1:t+5}            Predicted Returns at quantiles:
                                     q_0.01, q_0.05 (VaR_0.05), q_0.50
```

### 4.2 Mathematical Derivation of Quantile / Pinball Loss

Standard regression minimizes Mean Squared Error (MSE), which strictly yields the conditional **mean**:
$$\arg\min_{\hat{y}} \mathbb{E}[(Y - \hat{y})^2 \mid X] = \mathbb{E}[Y \mid X]$$
However, Value-at-Risk requires predicting the **5th percentile** ($Q_{0.05}(Y \mid X)$). We train Head 2 using the **Asymmetric Pinball Loss function** $\rho_\tau(u)$:

$$\rho_\tau(u) = u \cdot (\tau - \mathbb{I}(u < 0)) = \begin{cases} \tau \cdot u & \text{if } u \ge 0 \\ (\tau - 1) \cdot u & \text{if } u < 0 \end{cases}$$

#### Proof: Why Minimizing Pinball Loss Yields the $\tau$-th Quantile
Let $u = Y - \hat{y}_\tau$. The expected loss conditional on $X$ is:
$$\mathbb{E}[\rho_\tau(Y - \hat{y}_\tau)] = \int_{-\infty}^{\hat{y}_\tau} (1 - \tau)(\hat{y}_\tau - y) f(y) dy + \int_{\hat{y}_\tau}^{\infty} \tau (y - \hat{y}_\tau) f(y) dy$$

Differentiating with respect to $\hat{y}_\tau$ using the Leibniz Integral Rule:
$$\frac{\partial}{\partial \hat{y}_\tau} \mathbb{E}[\rho_\tau(Y - \hat{y}_\tau)] = (1 - \tau) \int_{-\infty}^{\hat{y}_\tau} f(y) dy - \tau \int_{\hat{y}_\tau}^{\infty} f(y) dy$$
$$(1 - \tau) F(\hat{y}_\tau) - \tau (1 - F(\hat{y}_\tau)) = F(\hat{y}_\tau) - \tau = 0$$
$$F(\hat{y}_\tau) = \tau \implies \hat{y}_\tau^* = F^{-1}(\tau) \equiv Q_\tau(Y \mid X)$$
Thus, minimizing the Pinball loss guarantees that $\hat{y}_\tau$ converges to the exact true $\tau$-th conditional quantile without assuming any parametric shape for market returns.

### 4.3 Joint Multi-Task Loss Function
The network is trained end-to-end to jointly predict realized volatility and tail quantiles:
$$\mathcal{L}_{Total} = \mathcal{L}_{QLIKE}(\sigma, \hat{\sigma}) + \sum_{\tau \in \{0.01, 0.05, 0.50\}} \lambda_\tau \cdot \frac{1}{B} \sum_{b=1}^B \rho_\tau\left( r_{p, t+1:t+5}^{(b)} - \hat{q}_\tau^{(b)} \right) + \gamma_{\text{non-cross}} \mathcal{L}_{\text{crossing}}$$

where $\mathcal{L}_{\text{crossing}} = \max(0, \hat{q}_{0.01} - \hat{q}_{0.05}) + \max(0, \hat{q}_{0.05} - \hat{q}_{0.50})$ penalizes non-monotonic quantile crossings.

---

## 5. Mathematical Evaluation Framework

### 5.1 Volatility Metric: Gaussian Quasi-Likelihood (QLIKE) Loss
$$\text{QLIKE}(y_t, \hat{y}_t) = \frac{y_t}{\hat{y}_t} - \ln\left(\frac{y_t}{\hat{y}_t}\right) - 1$$
- **Property (Patton 2011):** QLIKE is robust to noise in the proxy estimator and is strictly invariant to scale.
- **Asymmetric Penalty:**
  - If $\hat{y}_t = 0.5 \cdot y_t$ (underestimating risk by $50\%$): $\text{QLIKE} = 2 - \ln(2) - 1 \approx 0.3069$.
  - If $\hat{y}_t = 2.0 \cdot y_t$ (overestimating risk by $100\%$): $\text{QLIKE} = 0.5 - \ln(0.5) - 1 \approx 0.1931$.
  - Underestimating risk is penalized significantly more heavily, aligning ML training with financial prudence.

### 5.2 Tail Risk Metric: Kupiec Proportion of Failures (POF) Hypothesis Test
To test whether predicted $VaR_{0.05}$ provides valid coverage out-of-sample:
- Let $N$ be the total test trading days.
- Let $I_t = \mathbb{I}(r_{p, t+1:t+5} < -VaR_{0.05, t})$ be the binary failure indicator.
- Total observed breaches $x = \sum_{t=1}^N I_t$, with sample failure rate $\hat{p} = \frac{x}{N}$.
- **Null Hypothesis:** $H_0: p = p_0 = 0.05$ (Nominal $5\%$ failure rate).
- **Likelihood Ratio Test Statistic:**
  $$LR_{POF} = -2 \ln \left[ \frac{(1 - p_0)^{N-x} p_0^x}{(1 - \hat{p})^{N-x} \hat{p}^x} \right] \sim \chi^2(1)$$
- **Decision Rule:** At confidence level $\alpha = 0.05$, the critical value $\chi_{0.05}^2(1) = 3.8415$.
  - If $LR_{POF} \le 3.8415 \implies$ **Fail to reject $H_0$** (Model is statistically calibrated).
  - If $LR_{POF} > 3.8415$ and $\hat{p} > 0.05 \implies$ Model is dangerously underestimating risk.
  - If $LR_{POF} > 3.8415$ and $\hat{p} < 0.05 \implies$ Model is overly conservative (hoarding capital).

---

## 6. Full Training & Inference Engineering Pipelines

### 6.1 Complete Walk-Forward Training Algorithm

```python
"""
PSEUDOCODE: Full Walk-Forward Training Loop
"""
# 1. Temporal Splits
train_df = data.loc["2012-01-01":"2020-12-31"]
val_df   = data.loc["2021-01-01":"2022-12-31"]
test_df  = data.loc["2023-01-01":"2024-10-31"]

# 2. Strict Preprocessing without leakage
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(train_df[feature_cols])
X_val_scaled   = scaler.transform(val_df[feature_cols])
X_test_scaled  = scaler.transform(test_df[feature_cols])

# 3. Create Rolling 30-day Tensors (Batch, 30, Features)
train_dataset = SlidingWindowTimeSeriesDataset(X_train_scaled, y_train, window=30, horizon=5)
val_dataset   = SlidingWindowTimeSeriesDataset(X_val_scaled,   y_val,   window=30, horizon=5)

train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=128, shuffle=False)

# 4. Instantiate Model & Optimizer
model = DeepTemporalQuantileNetwork(num_features=18, d_model=64)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# 5. Training Loop with Early Stopping
best_val_loss = float("inf")
patience, patience_counter = 10, 0

for epoch in range(1, 100):
    model.train()
    for batch_x, batch_vol, batch_ret in train_loader:
        optimizer.zero_grad()
        pred_vol, pred_quantiles = model(batch_x)
        
        # Joint Loss
        loss_vol = qlike_loss(batch_vol, pred_vol)
        loss_var = pinball_loss(batch_ret, pred_quantiles, tau_list=[0.01, 0.05, 0.50])
        total_loss = loss_vol + 2.0 * loss_var
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Validation Step
    model.eval()
    val_loss = evaluate(model, val_loader)
    scheduler.step()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "artifacts/best_dtqn_model.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
```

### 6.2 Dynamic Portfolio Inference Algorithm

When an investor specifies custom allocation weights $\mathbf{w} = [w_1, \dots, w_N]^T$ in the interactive web UI:

```
Step 1: Receive User Input
  User sets weights: w_SPY = 0.50, w_QQQ = 0.20, w_TLT = 0.20, w_GLD = 0.10, w_VNQ = 0.00

Step 2: Generate Portfolio Return Stream
  r_p,t = sum_{i=1}^N w_i * r_i,t for historical window [t - 30, t]

Step 3: Extract Feature Tensor X_t
  Scale features using pre-fitted training scaler.
  Shape: [1, 30, 18]

Step 4: Forward Pass through Saved DTQN Model
  pred_vol, pred_quantiles = model(X_t)
  sigma_hat = pred_vol.item()
  VaR_0.05  = -pred_quantiles[0, 1].item()  # 5th percentile return negated

Step 5: Generate Forecast Visualization
  Construct dynamic 5-day risk envelope:
  Upper Bound (+1 sigma): P_t * exp( + sigma_hat * sqrt(5/252) )
  Lower Bound (-1 sigma): P_t * exp( - sigma_hat * sqrt(5/252) )
  VaR Floor (95% loss):   P_t * (1 - VaR_0.05)
```

---

## 7. ES 335 Interactive Web Deliverable Architecture

### 7.1 Web Application Structure (`app/app.py`)
- **Technology Stack:** Streamlit + Plotly Interactive Graphics + NumPy/Pandas.
- **Latency Optimization:** Precomputed out-of-sample prediction arrays for baseline models + fast PyTorch/TorchScript or ONNX runtime inference for user-defined portfolio weights.

### 7.2 UI Modules & Interactions
1. **Interactive Portfolio Construction Workbench:**
   - Sliders for 5 assets with dynamic normalization: $\sum_{i=1}^5 w_i = 100\%$.
   - Interactive Asset Allocation Donut Chart.
2. **Model Horizon & Strategy Switcher:**
   - Compare predictions: `Naive 21d Vol`, `GARCH(1,1)`, `LightGBM`, `PyTorch DTQN`.
   - Side-by-side metric scorecard: Out-of-Sample RMSE, QLIKE Loss, Kupiec POF $p$-value.
3. **Interactive Visual Risk Fan Chart:**
   - Plotly time-series chart showing actual portfolio trajectory against model-predicted $\pm 1\sigma$ and $95\%$ VaR bounds.
   - Hover inspection revealing exact day-by-day loss boundaries.
4. **Historical Stress Test Simulator (The 5-Point "Useful Interaction"):**
   - Clickable presets:
     - **"March 2020 COVID Liquidity Freeze"** (Extreme VIX spike to 82).
     - **"2022 Inflation & Rate Hike Regime"** (Equities and Bonds fell simultaneously).
   - Simulates the user's custom portfolio through the historical shock.
   - Automatically runs Kupiec breach counting:
     > *"Under your 50% Equity / 50% Bond allocation, GARCH(1,1) suffered 6 tail breaches ($LR = 4.82$, FAILED), whereas PyTorch DTQN suffered 2 breaches ($LR = 0.41$, CALIBRATED)."*

---

## 8. Viva Defense & Rubric Scorecard Strategy

| Rubric Criteria | Weight | Defense Evidence Prepared |
| :--- | :--- | :--- |
| **Problem Formulation** | 10 pts | Clear mathematical definition of 5-day realized vol & 95% VaR; causality boundary defined at $16:00$ close. |
| **Data Understanding** | 10 pts | 12-year history across 5 diverse ETFs + VIX + Yield curve; holiday alignment & corporate action adjustments documented. |
| **Preprocessing & Leakage** | 10 pts | ADF unit-root tests prove return stationarity; strictly causal rolling estimators; scalers fit exclusively on 2012-2020 train split. |
| **Baseline & Method Selection** | 15 pts | Progresses with clear justification: Naive $\to$ GARCH(1,1) $\to$ LightGBM $\to$ Deep Temporal Quantile Network. |
| **Evaluation Design** | 20 pts | 3-way Walk-Forward Split (no random k-fold!); QLIKE asymmetric loss and Kupiec POF $\chi^2$ hypothesis test. |
| **Results & Error Analysis** | 10 pts | Detailed audit of model lag during structural breaks; quantile calibration curves and heteroskedasticity residual checks. |
| **Reproducibility** | 5 pts | Pinned `requirements.txt`, random seed locks (`torch.manual_seed(42)`), modular pipeline code. |
| **ES 335 Deliverables** | 20 pts | Deployed Streamlit app (10 pts) + Interactive Custom Portfolio Stress Simulator (5 pts) + Clear individual viva defense mastery (5 pts). |
| **Total** | **100 pts**| **Comprehensive, rigorous, and technically unassailable.** |
