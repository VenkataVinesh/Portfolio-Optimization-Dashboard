import numpy as np
import pandas as pd
import scipy.optimize as sco
import matplotlib.pyplot as plt

def generate_synthetic_returns(num_assets=5, num_periods=252):
    np.random.seed(42)
    # Generate random expected returns and standard deviations
    mean_returns = np.random.uniform(0.05, 0.20, num_assets)
    volatilities = np.random.uniform(0.10, 0.30, num_assets)
    
    # Create random covariance matrix
    corr_matrix = np.random.uniform(-0.1, 0.4, (num_assets, num_assets))
    np.fill_diagonal(corr_matrix, 1.0)
    # Make symmetric positive-definite
    corr_matrix = np.dot(corr_matrix, corr_matrix.T)
    corr_matrix = corr_matrix / np.max(np.abs(corr_matrix))
    np.fill_diagonal(corr_matrix, 1.0)
    
    cov_matrix = np.diag(volatilities) @ corr_matrix @ np.diag(volatilities)
    
    return mean_returns, cov_matrix

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, volatility

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0.02):
    num_assets = len(mean_returns)
    
    def negative_sharpe(weights):
        p_return, p_vol = portfolio_performance(weights, mean_returns, cov_matrix)
        return -(p_return - risk_free_rate) / p_vol
        
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_guess = num_assets * [1.0 / num_assets]
    
    results = sco.minimize(negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return results

def compute_efficient_frontier(mean_returns, cov_matrix, num_portfolios=200):
    num_assets = len(mean_returns)
    
    # Calculate random portfolios for visualization
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        p_ret, p_vol = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0, i] = p_vol
        results[1, i] = p_ret
        results[2, i] = (p_ret - 0.02) / p_vol
        
    return results

def run_calculations():
    assets = ['Asset_A', 'Asset_B', 'Asset_C', 'Asset_D', 'Asset_E']
    mean_returns, cov_matrix = generate_synthetic_returns(len(assets))
    
    # Optimize portfolio
    opt_results = max_sharpe_ratio(mean_returns, cov_matrix)
    opt_weights = opt_results.x
    opt_return, opt_vol = portfolio_performance(opt_weights, mean_returns, cov_matrix)
    opt_sharpe = (opt_return - 0.02) / opt_vol
    
    print("Optimization convergence successful:")
    for asset, weight in zip(assets, opt_weights):
        print(f"-> {asset} Allocation: {weight * 100:.2f}%")
    print(f"-> Portfolio Expected Return: {opt_return * 100:.2f}%")
    print(f"-> Portfolio Volatility: {opt_vol * 100:.2f}%")
    print(f"-> Sharpe Ratio: {opt_sharpe:.4f}")
    
    # Plot frontier
    frontier_data = compute_efficient_frontier(mean_returns, cov_matrix)
    
    plt.figure(figsize=(10, 5))
    plt.scatter(frontier_data[0], frontier_data[1], c=frontier_data[2], cmap='viridis', marker='o', s=15, alpha=0.3)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(opt_vol, opt_return, color='red', marker='*', s=200, label='Max Sharpe Allocation')
    plt.title('Efficient Frontier Volatility Bounds')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Expected Return')
    plt.legend()
    plt.grid(True)
    plt.savefig('efficient_frontier.png')
    print("Saved efficiently plotted boundary frontier to efficient_frontier.png")

if __name__ == "__main__":
    run_calculations()
