# Portfolio Optimization via Stochastic Portfolio Theory (SPT)

A research-level implementation of multiple portfolio optimization strategies on Indian equity data (Nifty 50 universe), progressing from classical baselines to neural network-driven SPT models inspired by **Kom Samo & Vervuurt (2016)**.

---

## Methods Implemented

| Method | Type | Loss Function |
|---|---|---|
| Equal Weight | Baseline | No optimization |
| Static Markowitz | Classical | Max Sharpe (analytical) |
| Rolling Markowitz | Classical | Max Sharpe (rolling window) |
| ML Portfolio (SPTModel) | Neural Network | Sharpe + Drawdown + L2 |
| Classical SPT (Diversity Weighted) | Analytical | No optimization |
| Parametric SPT | Learned scalar p | Relative Sharpe Loss |
| Functional SPT | Deep Network + Residual | Relative Log-Wealth + Sharpe + Downside + Concentration |

---

## Loss Functions

### 1. Equal Weight Portfolio

No loss function. Weights are fixed:

$$w_i = \frac{1}{N} \quad \forall i$$

Serves as a naive diversification baseline.

---

### 2. Static Markowitz (Max Sharpe)

$$\max_{w} \frac{\mu^\top w - r_f}{\sqrt{w^\top \Sigma w}}$$

**where:**
- $\mu$ = EMA historical expected returns
- $\Sigma$ = Sample covariance matrix
- $w$ = Portfolio weights
- $r_f$ = Risk-free rate (default 0)

**Constraints:** $\sum w_i = 1$, $\quad 0 \leq w_i \leq 0.4$

Solved analytically via PyPortfolioOpt (`EfficientFrontier`).

---

### 3. Rolling Markowitz (Out-of-Sample)

Same objective as Static Markowitz but estimated on a rolling window:

$$\max_{w} \frac{\mu_{\text{window}}^\top w - r_f}{\sqrt{w^\top \Sigma_{\text{window}} w}}$$

- Training window: **252 days**
- Rebalance frequency: **every 21 days**
- $\mu_{\text{window}}$ and $\Sigma_{\text{window}}$ are re-estimated from the most recent 252 trading days only

---

### 4. ML Portfolio (SPTModel) — Neural Network

$$\mathcal{L} = -\frac{\mathbb{E}[R_p]}{\sigma(R_p) + \varepsilon} + \lambda_1 \sum_{\theta \in \text{model}} \|\theta\|^2 + \lambda_2 \cdot \text{MaxDrawdown}$$

**where:**

$$R_p = \sum_i w_i \cdot r_i \quad \text{(portfolio return per day)}$$

$$\text{MaxDrawdown} = \max_t \frac{\text{CumMax}_t - \text{Cum}_t}{\text{CumMax}_t}, \quad \text{Cum}_t = \prod_{s \leq t}(1 + R_p^s)$$

**Hyperparameters:**
- $\lambda_1 = 10^{-4}$ (L2 regularization)
- $\lambda_2 = 0.1$ (drawdown penalty)

**Model:** 3-layer MLP with BatchNorm + Dropout $\rightarrow$ Input $\to$ 64 $\to$ 32 $\to$ 1 $\to$ Softmax

---

### 5. Classical SPT — Diversity Weighted Portfolio

No loss function — closed-form analytical solution:

$$w_i = \frac{\mu_i^p}{\sum_j \mu_j^p}$$

**where:**
- $\mu_i$ = market weight of stock $i$ (price-based proxy)
- $p \in (0, 1)$ = diversity parameter

Tested with $p = 0.5$ and $p = 0.8$. Lower $p$ $\rightarrow$ more diversification away from large caps.

---

### 6. Parametric SPT — Learned p

$$\mathcal{L} = -\frac{\mathbb{E}[R_p - R_m]}{\sigma(R_p - R_m) + \varepsilon}$$

**where:**

$$R_p = \sum_i w_i(p) \cdot r_i, \qquad R_m = \sum_i \mu_i \cdot r_i$$

$$w_i(p) = \frac{\mu_i^p}{\sum_j \mu_j^p}$$

$p$ is a **learnable scalar parameter** initialized at $0.5$. The model learns the optimal diversity exponent by maximizing relative Sharpe over the market portfolio.

**Training:** Adam optimizer ($\text{lr} = 0.05$), 300 epochs

---

### 7. Functional SPT — Deep Network (Kom Samo & Vervuurt Inspired)

$$\mathcal{L} = -\mathcal{J} + \lambda_s \cdot \mathcal{P}_{\text{smooth}}$$

**Combined Objective:**

$$\mathcal{J} = \underbrace{\mathbb{E}\left[\log(1+R_p) - \log(1+R_m)\right]}_{\text{(1) Relative Log-Wealth}} + \lambda_{\text{sharpe}} \cdot \underbrace{\frac{\mathbb{E}[R_p]}{\sigma(R_p) + \varepsilon}}_{\text{(2) Sharpe}} - \gamma \cdot \underbrace{\mathbb{E}\left[\min(R_p, 0)^2\right]}_{\text{(3) Downside}} - 0.1 \cdot \underbrace{\mathbb{E}\left[\sum_i w_i^2\right]}_{\text{(4) Concentration}}$$

**Smoothness Penalty:**

$$\mathcal{P}_{\text{smooth}} = \mathbb{E}\left[(\Delta^2 f)^2\right], \quad \Delta^2 f = \Delta f_{t+1} - \Delta f_t, \quad \Delta f_t = f(x_{t+1}) - f(x_t)$$

**Term Explanations:**

| Term | Formula | Purpose |
|---|---|---|
| Relative Log-Wealth | $\mathbb{E}[\log(1+R_p) - \log(1+R_m)]$ | Maximize compounded growth over market |
| Sharpe | $\mathbb{E}[R_p] / \sigma(R_p)$ | Reward risk-adjusted returns |
| Downside | $\mathbb{E}[\min(R_p, 0)^2]$ | Penalize losses quadratically |
| Concentration | $\mathbb{E}[\sum_i w_i^2]$ | Force diversification (Herfindahl Index) |
| Smoothness | $\mathbb{E}[(\Delta^2 f)^2]$ | Smooth weight function across ranks |

**Hyperparameters:**
- $\lambda_{\text{sharpe}} = 0.5$
- $\gamma = 3.0$ (downside weight)
- $\lambda_s = 3.0$ (smoothness weight)

**Model Architecture:**

$$x_{\text{rank}} \xrightarrow{\text{Linear}(1 \to 32)} \xrightarrow{\text{Tanh}} \xrightarrow{+\text{ Residual}} \xrightarrow{\text{Linear}(32 \to 16)} \xrightarrow{\text{Tanh}} \xrightarrow{\text{Linear}(16 \to 1)} \xrightarrow{\text{Softplus}} \xrightarrow{\text{Normalize}} w_i$$

**Training:**
- Optimizer: Adam ($\text{lr} = 0.001$)
- Scheduler: CosineAnnealingWarmRestarts ($T_0 = 200$, $T_{\text{mult}} = 2$)
- Gradient Clipping: $\text{max\_norm} = 1.0$
- Early Stopping: patience = 200
- Max Epochs: 2000

---

## Results Summary

### With Volatility Targeting + Drawdown Control

| Method | Sharpe | Arithmetic Return | Geometric Return | Volatility |
|---|---|---|---|---|
| Static Markowitz | **1.3066** | 19.84% | 18.68% | 15.18% |
| ML Portfolio | 1.1697 | 17.32% | 16.22% | 14.80% |
| Market | 1.1086 | 16.70% | 15.57% | 15.07% |
| Equal Weight | 1.1019 | 16.55% | 15.42% | 15.02% |
| Rolling Markowitz | 1.0664 | 16.08% | 14.94% | 15.08% |

### Without Real-World Constraints (Test Period Only)

| Method | Sharpe | Arithmetic Return | Geometric Return | Volatility |
|---|---|---|---|---|
| Static Markowitz | **1.2368** | 24.13% | 22.22% | 19.51% |
| Equal Weight | 0.9989 | 13.82% | 12.86% | 13.83% |
| Rolling Markowitz | 0.9378 | 15.31% | 13.97% | 16.33% |
| Market | 0.9289 | 12.94% | 11.97% | 13.93% |
| ML Portfolio | 0.8733 | 13.60% | 12.38% | 15.57% |

### Functional SPT (After Transaction Cost)

| Method | Sharpe | Max Drawdown |
|---|---|---|
| Functional SPT (Gross) | **1.1760** | -37.12% |
| Functional SPT (After Cost) | 1.1198 | -37.19% |
| Market | 0.7167 | -34.83% |

---

## Project Structure

```
Portfolio_Management/
│
├── Portfolio_Management.ipynb   # Main notebook
└── README.md
```

---

## Requirements

```
yfinance
pandas
numpy
torch
matplotlib
scikit-learn
pypfopt
```

Install all:
```bash
pip install yfinance pandas numpy torch matplotlib scikit-learn pypfopt
```

---

## References

- Kom Samo, Y. L., & Vervuurt, A. (2016). *Stochastic Portfolio Theory: A Machine Learning Perspective*
- Fernholz, E. R. (2002). *Stochastic Portfolio Theory*. Springer
- PyPortfolioOpt: [https://pyportfolioopt.readthedocs.io](https://pyportfolioopt.readthedocs.io)

---

## Author

**Harsh Mangukiya**
B.Tech ICT — Dhirubhai Ambani University
[github.com/harshmangukiya71](https://github.com/harshmangukiya71)
