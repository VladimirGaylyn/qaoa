import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

def load_results(results_dir: str = "warmstart_results") -> Dict:
    """Load all results from the results directory"""
    
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found!")
        return None
    
    # Load aggregate results if available
    aggregate_file = os.path.join(results_dir, 'aggregate_results.json')
    if os.path.exists(aggregate_file):
        with open(aggregate_file, 'r') as f:
            aggregate_data = json.load(f)
    else:
        aggregate_data = None
    
    # Load individual run results
    individual_results = []
    for filename in sorted(os.listdir(results_dir)):
        if filename.startswith('run_') and filename.endswith('.json'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                result = json.load(f)
                # Extract run number and risk factor from filename
                parts = filename.replace('.json', '').split('_')
                run_id = int(parts[1])
                risk_factor = float(parts[3])
                result['run_id'] = run_id
                result['risk_factor'] = risk_factor
                individual_results.append(result)
    
    return {
        'aggregate': aggregate_data,
        'individual': individual_results
    }

def create_performance_dashboard(results: Dict):
    """Create a comprehensive performance dashboard"""
    
    if not results or not results['individual']:
        print("No results to plot!")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Extract data for analysis
    df_list = []
    for result in results['individual']:
        df_list.append({
            'run_id': result['run_id'],
            'risk_factor': result['risk_factor'],
            'warmstart_sharpe': result['warmstart']['metrics']['sharpe'],
            'standard_sharpe': result['standard']['metrics']['sharpe'],
            'warmstart_return': result['warmstart']['metrics']['return'],
            'standard_return': result['standard']['metrics']['return'],
            'warmstart_risk': result['warmstart']['metrics']['risk'],
            'standard_risk': result['standard']['metrics']['risk'],
            'speedup': result['improvement']['speedup'],
            'value_improvement': result['improvement']['value_improvement'],
            'warmstart_time': result['warmstart']['time'],
            'standard_time': result['standard']['time']
        })
    
    df = pd.DataFrame(df_list)
    
    # 1. Sharpe Ratio Comparison by Risk Factor
    ax1 = fig.add_subplot(gs[0, 0])
    df_melted = df[['risk_factor', 'warmstart_sharpe', 'standard_sharpe']].melt(
        id_vars=['risk_factor'], var_name='Method', value_name='Sharpe Ratio'
    )
    df_melted['Method'] = df_melted['Method'].map({
        'warmstart_sharpe': 'Warmstart',
        'standard_sharpe': 'Standard'
    })
    sns.boxplot(data=df_melted, x='risk_factor', y='Sharpe Ratio', hue='Method', ax=ax1)
    ax1.set_title('Sharpe Ratio by Risk Factor')
    ax1.set_xlabel('Risk Factor')
    ax1.grid(True, alpha=0.3)
    
    # 2. Average Performance Metrics
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = ['Sharpe Ratio', 'Return', 'Risk']
    warmstart_avg = [df['warmstart_sharpe'].mean(), df['warmstart_return'].mean(), df['warmstart_risk'].mean()]
    standard_avg = [df['standard_sharpe'].mean(), df['standard_return'].mean(), df['standard_risk'].mean()]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax2.bar(x - width/2, warmstart_avg, width, label='Warmstart', color='green', alpha=0.7)
    ax2.bar(x + width/2, standard_avg, width, label='Standard', color='blue', alpha=0.7)
    ax2.set_xlabel('Metric')
    ax2.set_ylabel('Value')
    ax2.set_title('Average Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Speedup Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df['speedup'], bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax3.axvline(df['speedup'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {df["speedup"].mean():.2f}x')
    ax3.set_xlabel('Speedup Factor')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Speedup Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Value Improvement Distribution
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.hist(df['value_improvement'], bins=20, edgecolor='black', alpha=0.7, color='purple')
    ax4.axvline(df['value_improvement'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {df["value_improvement"].mean():.1f}%')
    ax4.set_xlabel('Value Improvement (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Objective Value Improvement')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance Over Runs
    ax5 = fig.add_subplot(gs[1, :2])
    for rf in df['risk_factor'].unique():
        df_rf = df[df['risk_factor'] == rf].sort_values('run_id')
        ax5.plot(df_rf['run_id'], df_rf['warmstart_sharpe'], 
                marker='o', label=f'Warmstart (RF={rf})', alpha=0.7)
        ax5.plot(df_rf['run_id'], df_rf['standard_sharpe'], 
                marker='s', linestyle='--', label=f'Standard (RF={rf})', alpha=0.5)
    ax5.set_xlabel('Run ID')
    ax5.set_ylabel('Sharpe Ratio')
    ax5.set_title('Sharpe Ratio Across All Runs')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # 6. Risk-Return Scatter
    ax6 = fig.add_subplot(gs[1, 2:])
    scatter1 = ax6.scatter(df['warmstart_risk'], df['warmstart_return'], 
                          c=df['risk_factor'], cmap='Greens', s=100, alpha=0.7,
                          label='Warmstart', marker='o')
    scatter2 = ax6.scatter(df['standard_risk'], df['standard_return'], 
                          c=df['risk_factor'], cmap='Blues', s=100, alpha=0.7,
                          label='Standard', marker='^')
    ax6.set_xlabel('Risk (Std Dev)')
    ax6.set_ylabel('Expected Return')
    ax6.set_title('Risk-Return Profile')
    plt.colorbar(scatter1, ax=ax6, label='Risk Factor')
    
    # Add efficient frontier line (approximate)
    risks = np.linspace(df[['warmstart_risk', 'standard_risk']].min().min(),
                       df[['warmstart_risk', 'standard_risk']].max().max(), 50)
    returns = 0.02 + 1.2 * risks  # Approximate efficient frontier
    ax6.plot(risks, returns, 'r--', alpha=0.3, label='Approx. Efficient Frontier')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Execution Time Comparison
    ax7 = fig.add_subplot(gs[2, 0])
    time_data = pd.DataFrame({
        'Warmstart': df['warmstart_time'],
        'Standard': df['standard_time']
    })
    ax7.boxplot([time_data['Warmstart'], time_data['Standard']], 
                labels=['Warmstart', 'Standard'])
    ax7.set_ylabel('Execution Time (seconds)')
    ax7.set_title('Execution Time Comparison')
    ax7.grid(True, alpha=0.3)
    
    # 8. Success Rate by Risk Factor
    ax8 = fig.add_subplot(gs[2, 1])
    success_rates = []
    risk_factors = sorted(df['risk_factor'].unique())
    for rf in risk_factors:
        df_rf = df[df['risk_factor'] == rf]
        success_rate = (df_rf['value_improvement'] > 0).mean() * 100
        success_rates.append(success_rate)
    
    ax8.bar(risk_factors, success_rates, color='darkgreen', alpha=0.7, edgecolor='black')
    ax8.set_xlabel('Risk Factor')
    ax8.set_ylabel('Success Rate (%)')
    ax8.set_title('Warmstart Success Rate by Risk Factor')
    ax8.grid(True, alpha=0.3)
    
    # 9. Correlation Heatmap
    ax9 = fig.add_subplot(gs[2, 2:])
    corr_columns = ['warmstart_sharpe', 'standard_sharpe', 'warmstart_return', 
                    'standard_return', 'warmstart_risk', 'standard_risk', 
                    'speedup', 'value_improvement']
    corr_matrix = df[corr_columns].corr()
    
    # Rename columns for better display
    rename_dict = {
        'warmstart_sharpe': 'W-Sharpe',
        'standard_sharpe': 'S-Sharpe',
        'warmstart_return': 'W-Return',
        'standard_return': 'S-Return',
        'warmstart_risk': 'W-Risk',
        'standard_risk': 'S-Risk',
        'speedup': 'Speedup',
        'value_improvement': 'Value Imp.'
    }
    corr_matrix = corr_matrix.rename(columns=rename_dict, index=rename_dict)
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax9, cbar_kws={'label': 'Correlation'})
    ax9.set_title('Correlation Matrix')
    
    plt.suptitle('Warmstart QAOA Performance Dashboard', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join('warmstart_results', 'performance_dashboard.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Dashboard saved to {output_file}")
    
    return fig

def create_portfolio_analysis(results: Dict):
    """Analyze portfolio composition across runs"""
    
    if not results or not results['individual']:
        print("No results to analyze!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Track asset selection frequency
    asset_frequency = {}
    
    for result in results['individual']:
        for asset in result['warmstart']['selected_assets']:
            if asset not in asset_frequency:
                asset_frequency[asset] = {'warmstart': 0, 'standard': 0}
            asset_frequency[asset]['warmstart'] += 1
        
        for asset in result['standard']['selected_assets']:
            if asset not in asset_frequency:
                asset_frequency[asset] = {'warmstart': 0, 'standard': 0}
            asset_frequency[asset]['standard'] += 1
    
    # 1. Top selected assets
    ax = axes[0, 0]
    top_assets = sorted(asset_frequency.items(), 
                       key=lambda x: x[1]['warmstart'] + x[1]['standard'], 
                       reverse=True)[:15]
    
    assets = [a[0] for a in top_assets]
    warmstart_counts = [a[1]['warmstart'] for a in top_assets]
    standard_counts = [a[1]['standard'] for a in top_assets]
    
    x = np.arange(len(assets))
    width = 0.35
    ax.bar(x - width/2, warmstart_counts, width, label='Warmstart', color='green', alpha=0.7)
    ax.bar(x + width/2, standard_counts, width, label='Standard', color='blue', alpha=0.7)
    ax.set_xlabel('Asset')
    ax.set_ylabel('Selection Frequency')
    ax.set_title('Top 15 Most Selected Assets')
    ax.set_xticks(x)
    ax.set_xticklabels(assets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Portfolio diversity
    ax = axes[0, 1]
    warmstart_diversity = []
    standard_diversity = []
    
    for result in results['individual']:
        # Calculate diversity as number of unique assets across all selections
        warmstart_diversity.append(len(set(result['warmstart']['selected_assets'])))
        standard_diversity.append(len(set(result['standard']['selected_assets'])))
    
    ax.hist([warmstart_diversity, standard_diversity], bins=10, 
            label=['Warmstart', 'Standard'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Unique Assets')
    ax.set_ylabel('Frequency')
    ax.set_title('Portfolio Diversity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Overlap analysis
    ax = axes[1, 0]
    overlaps = []
    for result in results['individual']:
        warmstart_set = set(result['warmstart']['selected_assets'])
        standard_set = set(result['standard']['selected_assets'])
        overlap = len(warmstart_set & standard_set) / len(warmstart_set | standard_set) * 100
        overlaps.append(overlap)
    
    ax.hist(overlaps, bins=20, edgecolor='black', alpha=0.7, color='purple')
    ax.set_xlabel('Portfolio Overlap (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Overlap Between Warmstart and Standard Portfolios')
    ax.axvline(np.mean(overlaps), color='red', linestyle='--', 
               label=f'Mean: {np.mean(overlaps):.1f}%')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Performance by portfolio composition
    ax = axes[1, 1]
    
    # Group results by whether warmstart outperformed standard
    outperformed = []
    underperformed = []
    
    for result in results['individual']:
        if result['warmstart']['metrics']['sharpe'] > result['standard']['metrics']['sharpe']:
            outperformed.append(result['improvement']['value_improvement'])
        else:
            underperformed.append(result['improvement']['value_improvement'])
    
    if outperformed:
        ax.bar(0, len(outperformed), color='green', alpha=0.7, 
               label=f'Outperformed ({len(outperformed)} runs)')
    if underperformed:
        ax.bar(1, len(underperformed), color='red', alpha=0.7, 
               label=f'Underperformed ({len(underperformed)} runs)')
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Warmstart Better', 'Standard Better'])
    ax.set_ylabel('Number of Runs')
    ax.set_title('Warmstart vs Standard Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Portfolio Composition Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join('warmstart_results', 'portfolio_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Portfolio analysis saved to {output_file}")
    
    return fig

def main():
    print("Loading warmstart results...")
    results = load_results()
    
    if results:
        print(f"Loaded {len(results['individual'])} individual run results")
        
        # Create performance dashboard
        print("\nCreating performance dashboard...")
        create_performance_dashboard(results)
        
        # Create portfolio analysis
        print("\nCreating portfolio analysis...")
        create_portfolio_analysis(results)
        
        # Print summary statistics
        if results['aggregate'] and 'statistics' in results['aggregate']:
            stats = results['aggregate']['statistics']
            print("\n" + "="*60)
            print("SUMMARY STATISTICS")
            print("="*60)
            print(f"Average Warmstart Sharpe: {stats['warmstart_sharpe']['mean']:.3f}")
            print(f"Average Standard Sharpe: {stats['standard_sharpe']['mean']:.3f}")
            print(f"Average Speedup: {stats['speedup']['mean']:.2f}x")
            print(f"Success Rate: {stats['summary']['success_rate']:.1f}%")
            print("="*60)
        
        print("\nAll plots saved to warmstart_results/")
    else:
        print("No results found. Please run warmstart_statistical_analysis.py first.")

if __name__ == "__main__":
    main()