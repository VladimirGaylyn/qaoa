"""
Classical Portfolio Strategies for Advanced Warm Start
Multiple classical optimization strategies to provide better QAOA initialization
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class ClassicalPortfolioStrategies:
    """Multiple classical strategies for warm start initialization"""
    
    def __init__(self, n_assets: int, budget: int, risk_factor: float):
        self.n_assets = n_assets
        self.budget = budget
        self.risk_factor = risk_factor
    
    def greedy_sharpe_selection(self, expected_returns: np.ndarray, 
                                covariance: np.ndarray) -> np.ndarray:
        """Select assets with highest Sharpe ratios"""
        solution = np.zeros(self.n_assets)
        
        # Calculate individual Sharpe ratios
        sharpe_ratios = []
        for i in range(self.n_assets):
            volatility = np.sqrt(covariance[i, i])
            if volatility > 0:
                sharpe = expected_returns[i] / volatility
            else:
                sharpe = expected_returns[i]
            sharpe_ratios.append(sharpe)
        
        # Select top assets
        top_indices = np.argsort(sharpe_ratios)[-self.budget:]
        solution[top_indices] = 1
        
        return solution
    
    def minimum_variance_selection(self, covariance: np.ndarray) -> np.ndarray:
        """Select assets that minimize portfolio variance"""
        solution = np.zeros(self.n_assets)
        
        # Find assets with lowest variance
        variances = np.diag(covariance)
        
        # Consider correlations: avoid highly correlated assets
        selected = []
        remaining = list(range(self.n_assets))
        
        # Select first asset (lowest variance)
        first_asset = np.argmin(variances)
        selected.append(first_asset)
        remaining.remove(first_asset)
        
        # Iteratively select assets with low variance and low correlation
        while len(selected) < self.budget and remaining:
            min_score = float('inf')
            best_asset = None
            
            for asset in remaining:
                # Calculate average correlation with selected assets
                avg_corr = 0
                for sel in selected:
                    std_i = np.sqrt(covariance[asset, asset])
                    std_j = np.sqrt(covariance[sel, sel])
                    if std_i > 0 and std_j > 0:
                        corr = covariance[asset, sel] / (std_i * std_j)
                    else:
                        corr = 0
                    avg_corr += abs(corr)
                avg_corr /= len(selected)
                
                # Score combines low variance and low correlation
                score = variances[asset] * (1 + avg_corr)
                
                if score < min_score:
                    min_score = score
                    best_asset = asset
            
            if best_asset is not None:
                selected.append(best_asset)
                remaining.remove(best_asset)
        
        solution[selected] = 1
        return solution
    
    def maximum_diversification_selection(self, expected_returns: np.ndarray,
                                        covariance: np.ndarray) -> np.ndarray:
        """Maximum Diversification Ratio portfolio"""
        solution = np.zeros(self.n_assets)
        
        # Calculate correlation matrix
        std_devs = np.sqrt(np.diag(covariance))
        std_devs[std_devs == 0] = 1e-10  # Avoid division by zero
        correlation = covariance / np.outer(std_devs, std_devs)
        
        # Use greedy approach for diversification
        selected = []
        
        # Start with highest return asset
        first = np.argmax(expected_returns)
        selected.append(first)
        
        # Add assets that maximize diversification
        while len(selected) < self.budget:
            max_div_benefit = -float('inf')
            best_asset = None
            
            for i in range(self.n_assets):
                if i in selected:
                    continue
                
                # Calculate diversification benefit
                # Lower correlation with existing = higher benefit
                avg_corr = np.mean([abs(correlation[i, j]) for j in selected])
                div_benefit = expected_returns[i] * (1 - avg_corr)
                
                if div_benefit > max_div_benefit:
                    max_div_benefit = div_benefit
                    best_asset = i
            
            if best_asset is not None:
                selected.append(best_asset)
        
        solution[selected] = 1
        return solution
    
    def risk_parity_selection(self, covariance: np.ndarray) -> np.ndarray:
        """Risk Parity approach - equal risk contribution"""
        solution = np.zeros(self.n_assets)
        
        # Calculate inverse volatility weights
        volatilities = np.sqrt(np.diag(covariance))
        volatilities[volatilities == 0] = 1e-10  # Avoid division by zero
        inv_vol_weights = 1.0 / volatilities
        inv_vol_weights /= np.sum(inv_vol_weights)
        
        # Select assets with highest inverse volatility weights
        top_indices = np.argsort(inv_vol_weights)[-self.budget:]
        solution[top_indices] = 1
        
        return solution
    
    def momentum_based_selection(self, expected_returns: np.ndarray,
                                covariance: np.ndarray) -> np.ndarray:
        """Select based on momentum (using returns as proxy)"""
        solution = np.zeros(self.n_assets)
        
        # Combine momentum (returns) with risk adjustment
        volatilities = np.sqrt(np.diag(covariance))
        volatilities[volatilities == 0] = 1e-10
        momentum_scores = expected_returns / (1 + volatilities)
        
        # Add mean-reversion component (avoid extreme values)
        mean_return = np.mean(expected_returns)
        std_return = np.std(expected_returns)
        if std_return > 0:
            z_scores = (expected_returns - mean_return) / std_return
            # Penalize extreme z-scores (likely to revert)
            adjusted_scores = momentum_scores * np.exp(-0.5 * np.abs(z_scores))
        else:
            adjusted_scores = momentum_scores
        
        top_indices = np.argsort(adjusted_scores)[-self.budget:]
        solution[top_indices] = 1
        
        return solution
    
    def correlation_clustering_selection(self, expected_returns: np.ndarray,
                                        covariance: np.ndarray) -> np.ndarray:
        """Select assets from different correlation clusters"""
        solution = np.zeros(self.n_assets)
        
        # Calculate correlation matrix
        std_devs = np.sqrt(np.diag(covariance))
        std_devs[std_devs == 0] = 1e-10
        correlation = covariance / np.outer(std_devs, std_devs)
        
        # Simple clustering based on correlation
        n_clusters = min(3, self.budget)
        clusters = self._correlation_clustering(correlation, n_clusters)
        
        # Select best assets from each cluster
        selected = []
        assets_per_cluster = self.budget // n_clusters
        extra = self.budget % n_clusters
        
        for cluster_id in range(n_clusters):
            cluster_assets = clusters[cluster_id]
            
            if not cluster_assets:
                continue
            
            # Rank assets in cluster by Sharpe ratio
            cluster_sharpes = []
            for i in cluster_assets:
                if std_devs[i] > 0:
                    sharpe = expected_returns[i] / std_devs[i]
                else:
                    sharpe = expected_returns[i]
                cluster_sharpes.append((sharpe, i))
            cluster_sharpes.sort(reverse=True)
            
            # Select top assets from this cluster
            n_select = assets_per_cluster + (1 if cluster_id < extra else 0)
            for j in range(min(n_select, len(cluster_sharpes))):
                selected.append(cluster_sharpes[j][1])
        
        # Fill remaining slots if needed
        if len(selected) < self.budget:
            remaining = [i for i in range(self.n_assets) if i not in selected]
            sharpe_scores = []
            for i in remaining:
                if std_devs[i] > 0:
                    sharpe = expected_returns[i] / std_devs[i]
                else:
                    sharpe = expected_returns[i]
                sharpe_scores.append((sharpe, i))
            sharpe_scores.sort(reverse=True)
            
            for sharpe, idx in sharpe_scores[:self.budget - len(selected)]:
                selected.append(idx)
        
        solution[selected[:self.budget]] = 1
        return solution
    
    def _correlation_clustering(self, correlation: np.ndarray, 
                               n_clusters: int) -> Dict[int, List[int]]:
        """Simple correlation-based clustering"""
        # Initialize clusters
        clusters = {i: [] for i in range(n_clusters)}
        assigned = []
        
        # Seed each cluster with an asset
        for i in range(min(n_clusters, self.n_assets)):
            clusters[i].append(i)
            assigned.append(i)
        
        # Assign remaining assets to nearest cluster
        for i in range(self.n_assets):
            if i in assigned:
                continue
            
            # Find cluster with highest average correlation
            best_cluster = 0
            best_corr = -1
            
            for cluster_id, cluster_assets in clusters.items():
                if not cluster_assets:
                    best_cluster = cluster_id
                    break
                
                avg_corr = np.mean([abs(correlation[i, j]) for j in cluster_assets])
                if avg_corr > best_corr:
                    best_corr = avg_corr
                    best_cluster = cluster_id
            
            clusters[best_cluster].append(i)
        
        return clusters
    
    def get_all_solutions(self, expected_returns: np.ndarray,
                         covariance: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Generate all classical solutions"""
        
        solutions = {
            'sharpe': self.greedy_sharpe_selection(expected_returns, covariance),
            'min_variance': self.minimum_variance_selection(covariance),
            'max_diversification': self.maximum_diversification_selection(expected_returns, covariance),
            'risk_parity': self.risk_parity_selection(covariance),
            'momentum': self.momentum_based_selection(expected_returns, covariance),
            'correlation_clustering': self.correlation_clustering_selection(expected_returns, covariance)
        }
        
        # Evaluate each solution
        qualities = {}
        for name, solution in solutions.items():
            if np.sum(solution) == self.budget:
                weights = solution / self.budget
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights, np.dot(covariance, weights))
                portfolio_risk = np.sqrt(portfolio_variance)
                
                # Calculate objective value (return - risk_factor * variance)
                objective = portfolio_return - self.risk_factor * portfolio_variance
                
                # Calculate Sharpe ratio
                if portfolio_risk > 0:
                    sharpe = portfolio_return / portfolio_risk
                else:
                    sharpe = portfolio_return
                
                qualities[name] = objective
            else:
                qualities[name] = -float('inf')
        
        return solutions, qualities