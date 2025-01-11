# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 21:44:11 2025

@author: user
"""


#Participants
#3231 TW
#1216 TW
#00757 TW
#2882 TW
#1319 TW
#2308 TW
#2382 TW
#00881 TW
#2881 TW
#0050 TW



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns

class TaiwanPortfolioOptimizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.returns = None
        self.benchmark_returns = None
        
    def load_data(self):
        try:
            self.df = pd.read_excel(self.file_path, index_col='Date', parse_dates=True)
            returns = np.log(self.df/self.df.shift(1)).dropna()
            self.benchmark_returns = returns['0050 TW']
            self.returns = returns.drop('0050 TW', axis=1)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def calculate_metrics(self, weights):
        portfolio_returns = (self.returns * weights).sum(axis=1)
        tracking_diff = portfolio_returns - self.benchmark_returns
        tracking_error = np.sqrt(np.mean(tracking_diff**2))
        excess_return = portfolio_returns.mean() - self.benchmark_returns.mean()
        
        annualized_return = portfolio_returns.mean() * 252
        annualized_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
        
        return {
            'tracking_error': tracking_error,
            'excess_return': excess_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio
        }

    def verify_constraints(self, weights):
        total_weight = np.sum(weights)
        print("\n=== Weight Constraint Verification ===")
        print(f"Total Portfolio Weight: {total_weight:.6f} ({total_weight*100:.4f}%)")
        
        print("\nIndividual Asset Weights:")
        for asset, weight in zip(self.returns.columns, weights):
            print(f"{asset}: {weight:.6f} ({weight*100:.2f}%)")
        
        if abs(total_weight - 1.0) > 1e-6:
            print("\n⚠️ WARNING: Weights do not sum to exactly 100%!")
            print(f"Deviation from 100%: {abs(1-total_weight)*100:.6f}%")
        else:
            print("\n✓ VERIFIED: Weights sum to exactly 100%")

    def optimize_portfolio(self, tracking_weight=0.5, max_weight=0.3, min_weight=0.02):
        num_assets = len(self.returns.columns)
        
        def objective(weights):
            metrics = self.calculate_metrics(weights)
            return (tracking_weight * metrics['tracking_error'] - 
                   (1 - tracking_weight) * metrics['excess_return'])
        
        # Explicit constraint for 100% weight sum
        def weight_sum_constraint(weights):
            return np.sum(weights) - 1.0  # Must equal zero (sum = 100%)
        
        constraints = [
            {'type': 'eq', 'fun': weight_sum_constraint}
        ]
        
        bounds = [(min_weight, max_weight) for _ in range(num_assets)]
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0/num_assets] * num_assets)
        
        result = minimize(objective,
                         x0=initial_weights,
                         method='SLSQP',
                         bounds=tuple(bounds),
                         constraints=constraints)
        
        if not result.success:
            print(f"Optimization failed: {result.message}")
            return None
            
        return result.x

    def plot_results(self, weights):
        # Create results dataframe
        results_df = pd.DataFrame({
            'Asset': self.returns.columns,
            'Weight': weights,
            'Weight_Pct': weights * 100
        }).sort_values('Weight', ascending=False)
        
        # 1. Pie Chart
        plt.figure(figsize=(12, 8))
        plt.pie(results_df['Weight'], 
                labels=results_df['Asset'],
                autopct='%1.1f%%',
                startangle=90)
        plt.title('Optimal Portfolio Allocation')
        plt.axis('equal')
        plt.show()
        
        # 2. Performance Plot
        portfolio_returns = (self.returns * weights).sum(axis=1)
        cum_portfolio = (1 + portfolio_returns).cumprod()
        cum_benchmark = (1 + self.benchmark_returns).cumprod()
        
        plt.figure(figsize=(12, 6))
        plt.plot(cum_portfolio.index, cum_portfolio, label='Portfolio')
        plt.plot(cum_benchmark.index, cum_benchmark, label='0050 TW')
        plt.title('Cumulative Returns Comparison')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    file_path = r'D:\Research_2\New nine.xlsx'
    optimizer = TaiwanPortfolioOptimizer(file_path)
    
    if not optimizer.load_data():
        return
    
    print("=== Portfolio Optimization Parameters ===")
    print("1. All weights must sum to exactly 100%")
    print("2. Maximum weight per asset: 30%")
    print("3. Minimum weight per asset: 2%")
    print("4. Objective: Balance tracking error and excess return")
    
    # Optimize portfolio
    optimal_weights = optimizer.optimize_portfolio(
        tracking_weight=0.5,    # Equal weight between tracking error and excess return
        max_weight=0.30,        # Maximum 30% per asset
        min_weight=0.02         # Minimum 2% per asset
    )
    
    if optimal_weights is None:
        print("Portfolio optimization failed")
        return
    
    # Verify constraints
    optimizer.verify_constraints(optimal_weights)
    
    # Calculate and display metrics
    metrics = optimizer.calculate_metrics(optimal_weights)
    print("\n=== Portfolio Performance Metrics ===")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
    print(f"Tracking Error: {metrics['tracking_error']:.2%}")
    print(f"Excess Return: {metrics['excess_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Plot results
    optimizer.plot_results(optimal_weights)
    
    # Display final allocation table
    print("\n=== Final Portfolio Allocation ===")
    allocation_df = pd.DataFrame({
        'Asset': optimizer.returns.columns,
        'Weight': optimal_weights,
        'Allocation %': optimal_weights * 100
    }).sort_values('Weight', ascending=False)
    
    print(allocation_df.to_string(float_format=lambda x: '{:.4f}'.format(x)))

if __name__ == "__main__":
    main()