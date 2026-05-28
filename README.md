# Portfolio Optimization Dashboard

A Python project implementing Markowitz Mean-Variance Portfolio Optimization and Sharpe Ratio Maximization, built using SciPy and Pandas.

## Project Structure
```
Portfolio-Optimization-Dashboard/
├── optimization.py       # Markowitz optimization calculators
├── requirements.txt      # Numerical dependencies
└── README.md             # This setup guide
```

## Setup & Running

1. Install numerical dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run calculations (generates synthetic asset price returns, computes optimal risk-adjusted weights, and plots the Efficient Frontier):
   ```bash
   python optimization.py
   ```
3. The script will output the optimal asset allocation weights in your console and save `efficient_frontier.png`.

## Core Mathematical Models
- **Mean Portfolio Return:** $\mu_p = w^T \mu$
- **Portfolio Volatility:** $\sigma_p = \sqrt{w^T \Sigma w}$
- **Objective Function:** Minimize negative Sharpe Ratio:
  $$\min_{w} \quad - \frac{w^T \mu - R_f}{\sqrt{w^T \Sigma w}}$$
  subject to $\sum w_i = 1.0$ and $w_i \ge 0.0$ (no short selling).
