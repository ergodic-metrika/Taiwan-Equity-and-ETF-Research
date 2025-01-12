# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 16:40:49 2025

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

class RidgePortfolioOptimizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.returns = None
        self.benchmark_returns = None
        self.models = {}
        self.scaler = StandardScaler()
        self.lookback_period = 252  # One year of trading days
        
    def load_data(self):
        try:
            self.df = pd.read_excel(self.file_path, index_col='Date', parse_dates=True)
            returns = np.log(self.df/self.df.shift(1)).dropna()
            self.benchmark_returns = returns['0050 TW']
            self.returns = returns.drop('0050 TW', axis=1)
            print("Data loaded successfully")
            print(f"Date range: {self.returns.index[0]} to {self.returns.index[-1]}")
            print(f"Number of assets: {len(self.returns.columns)}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def create_features(self, returns_data):
        """Create features for prediction"""
        features = pd.DataFrame(index=returns_data.index)
        
        # Basic return features
        features['returns'] = returns_data
        
        # Lagged returns
        for i in range(1, 6):
            features[f'lag_return_{i}'] = returns_data.shift(i)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features[f'rolling_mean_{window}d'] = returns_data.rolling(window=window).mean()
            features[f'rolling_std_{window}d'] = returns_data.rolling(window=window).std()
            features[f'rolling_momentum_{window}d'] = returns_data.rolling(window=window).sum()
        
        # Volatility features
        for window in [5, 10, 20]:
            features[f'volatility_{window}d'] = returns_data.rolling(window=window).std()
            
        # Mean reversion features
        features['ma_5d'] = returns_data.rolling(window=5).mean()
        features['ma_20d'] = returns_data.rolling(window=20).mean()
        features['distance_to_ma_5d'] = returns_data - features['ma_5d']
        features['distance_to_ma_20d'] = returns_data - features['ma_20d']
        
        # Remove any infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        return features

    def train_ridge_models(self):
        """Train Ridge regression models for each asset"""
        print("\nTraining Ridge Regression Models...")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for asset in self.returns.columns:
            print(f"\nTraining model for {asset}")
            
            try:
                # Create features for this asset
                features = self.create_features(self.returns[asset])
                
                # Prepare target (next day's return)
                y = self.returns[asset].shift(-1)  # Next day's return
                
                # Align features and target
                data = pd.concat([features, y], axis=1)
                data = data.dropna()  # Remove any rows with NaN values
                
                # Split into X and y after alignment
                X = data.iloc[:, :-1]  # All columns except the last one
                y = data.iloc[:, -1]   # Last column is the target
                
                # Find optimal alpha using RidgeCV
                alphas = np.logspace(-6, 6, 13)
                ridge_cv = RidgeCV(
                    alphas=alphas,
                    cv=tscv,
                    scoring='neg_mean_squared_error'
                )
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Fit model
                ridge_cv.fit(X_scaled, y)
                
                # Store the best model
                best_alpha = ridge_cv.alpha_
                best_model = Ridge(alpha=best_alpha)
                best_model.fit(X_scaled, y)
                
                # Calculate performance metrics
                y_pred = best_model.predict(X_scaled)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                print(f"Best alpha: {best_alpha:.6f}")
                print(f"MSE: {mse:.6f}")
                print(f"R2 Score: {r2:.6f}")
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': np.abs(best_model.coef_)
                }).sort_values('Importance', ascending=False)
                
                print("\nTop 5 important features:")
                print(feature_importance.head())
                
                # Store model and scaler
                self.models[asset] = {
                    'model': best_model,
                    'scaler': self.scaler,
                    'features': X.columns,
                    'metrics': {
                        'mse': mse,
                        'r2': r2,
                        'alpha': best_alpha
                    }
                }
                
                # Plot actual vs predicted returns
                plt.figure(figsize=(12, 6))
                
                # Plot last year of data
                plt.subplot(1, 2, 1)
                plt.plot(y.index[-252:], y.values[-252:], label='Actual', alpha=0.7)
                plt.plot(y.index[-252:], y_pred[-252:], label='Predicted', alpha=0.7)
                plt.title(f'{asset}: Last Year Returns')
                plt.legend()
                plt.xticks(rotation=45)
                
                # Feature importance plot
                plt.subplot(1, 2, 2)
                top_features = feature_importance.head(10)
                plt.barh(top_features['Feature'], top_features['Importance'])
                plt.title('Top 10 Feature Importance')
                
                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"Error training model for {asset}: {str(e)}")
                continue

    def predict_returns(self, asset, features):
        """Predict returns using trained Ridge models"""
        try:
            model_info = self.models[asset]
            
            # Ensure features match training features
            required_features = model_info['features']
            features = features[required_features]
            
            # Scale features
            features_scaled = model_info['scaler'].transform(features)
            return model_info['model'].predict(features_scaled)
        except Exception as e:
            print(f"Error predicting returns for {asset}: {str(e)}")
            return np.zeros(len(features))

    def optimize_portfolio(self, tracking_weight=0.5, max_weight=0.3, min_weight=0.02):
        """Optimize portfolio weights using Ridge predictions"""
        print("\nOptimizing portfolio weights...")
        
        # Get latest features and predictions
        predicted_returns = {}
        for asset in self.returns.columns:
            features = self.create_features(self.returns[asset]).iloc[-1:]
            predicted_returns[asset] = self.predict_returns(asset, features)[0]
        
        def objective(weights):
            # Expected return component
            expected_return = sum(w * predicted_returns[asset] 
                                for asset, w in zip(self.returns.columns, weights))
            
            # Tracking error component
            portfolio_returns = (self.returns * weights).sum(axis=1)
            tracking_diff = portfolio_returns - self.benchmark_returns
            tracking_error = np.sqrt(np.mean(tracking_diff**2))
            
            # Combine objectives
            return -(expected_return - tracking_weight * tracking_error)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Sum of weights = 1
        ]
        
        # Bounds for individual weights
        bounds = [(min_weight, max_weight) for _ in range(len(self.returns.columns))]
        
        # Initial guess: equal weights
        x0 = np.array([1.0/len(self.returns.columns)] * len(self.returns.columns))
        
        # Optimize
        result = minimize(objective,
                         x0=x0,
                         method='SLSQP',
                         bounds=tuple(bounds),
                         constraints=constraints)
        
        if not result.success:
            print("Optimization failed:", result.message)
            return None
            
        print("Portfolio optimization completed successfully")
        return result.x

    def plot_results(self, weights):
        """Plot portfolio allocation and performance metrics"""
        
        # Create figure with subplots
        plt.figure(figsize=(15, 10))
        
        # 1. Portfolio Allocation Pie Chart
        plt.subplot(2, 2, 1)
        plt.pie(weights, 
                labels=self.returns.columns,
                autopct='%1.1f%%',
                startangle=90)
        plt.title('Portfolio Allocation')
        
        # 2. Bar Plot of Weights
        plt.subplot(2, 2, 2)
        plt.bar(self.returns.columns, weights)
        plt.title('Portfolio Weights')
        plt.xticks(rotation=45)
        
        # 3. Cumulative Returns Plot
        portfolio_returns = (self.returns * weights).sum(axis=1)
        cum_returns = (1 + portfolio_returns).cumprod()
        cum_benchmark = (1 + self.benchmark_returns).cumprod()
        
        plt.subplot(2, 1, 2)
        plt.plot(cum_returns.index, cum_returns, label='Portfolio')
        plt.plot(cum_benchmark.index, cum_benchmark, label='Benchmark (0050 TW)')
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print performance metrics
        print("\nPortfolio Performance Metrics:")
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0
        tracking_error = np.sqrt(np.mean((portfolio_returns - self.benchmark_returns)**2)) * np.sqrt(252)
        
        metrics = pd.DataFrame({
            'Metric': [
                'Annual Return',
                'Annual Volatility',
                'Sharpe Ratio',
                'Tracking Error',
                'Information Ratio'
            ],
            'Value': [
                f"{annual_return:.2%}",
                f"{annual_vol:.2%}",
                f"{sharpe_ratio:.2f}",
                f"{tracking_error:.2%}",
                f"{(annual_return / tracking_error if tracking_error != 0 else 0):.2f}"
            ]
        })
        
        print(metrics.to_string(index=False))

def main():
    file_path = r'D:\Research_2\New nine.xlsx'
    optimizer = RidgePortfolioOptimizer(file_path)
    
    if not optimizer.load_data():
        return
    
    # Train models
    optimizer.train_ridge_models()
    
    # Optimize portfolio
    optimal_weights = optimizer.optimize_portfolio(
        tracking_weight=0.5,
        max_weight=0.30,
        min_weight=0.02
    )
    
    if optimal_weights is not None:
        # Plot results
        optimizer.plot_results(optimal_weights)
        
        # Display final allocation
        print("\nFinal Portfolio Allocation:")
        allocation_df = pd.DataFrame({
            'Asset': optimizer.returns.columns,
            'Weight': optimal_weights,
            'Weight %': optimal_weights * 100
        }).sort_values('Weight', ascending=False)
        
        print(allocation_df.to_string(index=False, float_format=lambda x: '{:.4f}'.format(x)))

if __name__ == "__main__":
    main()