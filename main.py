import pandas as pd
import sys
import os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_pipeline import DataPipeline
from synthetic_pricer import SyntheticPricer
from statistical_models import StatisticalModels
from risk_analytics import RiskAnalytics
import config


def save_results(results: dict, output_dir: str = None):
    """Save DataFrames to CSV files"""
    output_dir = output_dir or config.PATH_CONFIG['results_path']
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved = 0
    
    for name, data in results.items():
        if isinstance(data, pd.DataFrame) and not data.empty:
            data.to_csv(f"{output_dir}/{name}_{timestamp}.csv")
            saved += 1
    
    print(f"Saved {saved} files to {output_dir}")


def print_summary(market_data, pricing_results, statistical_results, risk_results=None):
    """Print analysis summary"""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Market data
    m = market_data['metadata']
    print(f"\nMarket Data: {m['num_tickers']} tickers, {m['trading_days']} days")
    
    # Pricing
    p = pricing_results['summary']
    if 'avg_synthetic_cost' in p:
        print(f"\nPricing:")
        print(f"  Synthetic: ${p['avg_synthetic_cost']:,.0f}, Cash: ${p['avg_cash_long_cost']:,.0f}")
        print(f"  Savings: ${p['total_savings_vs_long']:,.0f}, Recs: {p['recommendations']}")
    
    # Statistical
    s = statistical_results['summary']
    print(f"\nStatistical: Models fitted: {s['models_fitted']}")
    if 'avg_r_squared' in s:
        print(f"  Avg R²: {s['avg_r_squared']:.3f}")
    if 'hypothesis_test' in s:
        h = s['hypothesis_test']
        sig = 'Yes' if h['significant'] else 'No'
        print(f"  Significant: {sig}, Mean savings: {h['mean_savings']*10000:.1f} bps (p={h['p_value']:.4f})")
    
    # Risk
    if risk_results:
        r = risk_results['portfolio_summary']
        mc = risk_results['monte_carlo']['portfolio']
        print(f"\nRisk (Portfolio ${r['total_value']:,.0f}):")
        print(f"  VaR: ${abs(r['var_95']):,.0f}, CVaR: ${abs(r['cvar_95']):,.0f}")
        print(f"  Sharpe: {r['sharpe_ratio']:.2f}, Sortino: {r['sortino_ratio']:.2f}, Max DD: {r['max_drawdown']:.2%}")
        print(f"  MC Projection: ${mc['mean_final']:,.0f} (worst 5%: ${mc['worst_5pct']:,.0f}, prob loss: {mc['prob_loss']:.1%})")
    
    # Signals
    signals = statistical_results.get('zscore_signals', {})
    if signals:
        print(f"\nTop 5 Signals:")
        for ticker, signal_df in list(signals.items())[:5]:
            if not signal_df.empty:
                current = signal_df.iloc[-1]
                print(f"  {ticker}: {current['signal']:<12} (z={current['zscore']:>6.2f})")
    
    print("="*60 + "\n")


def run_analysis(tickers: list = None,
                start_date: str = None,
                end_date: str = None,
                force_download: bool = False,
                save_output: bool = True,
                position_value: float = 1000000,
                run_risk_analysis: bool = True):
    """Run complete analysis pipeline"""
    
    print("\nSynthetic Spread Arbitrage Analysis")
    print("="*60)
    
    # Defaults
    tickers = tickers or config.DATA_CONFIG['tickers']
    start_date = start_date or config.DATA_CONFIG['start_date']
    end_date = end_date or config.DATA_CONFIG['end_date']
    
    print(f"Tickers: {len(tickers)}, Period: {start_date} to {end_date}")
    if run_risk_analysis:
        print(f"Risk analysis: ${position_value:,.0f} per ticker")
    print()
    
    try:
        # Step 1: Market data
        print("Step 1: Collecting market data...")
        pipeline = DataPipeline()
        market_data = pipeline.run_full_pipeline(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            force_download=force_download
        )
        
        # Step 2: Pricing
        print("\nStep 2: Running pricing analysis...")
        pricer = SyntheticPricer()
        pricing_results = pricer.run_full_pricing_analysis(
            data=market_data,
            days=90,
            notional=100000,
            tax_rate=0.30
        )
        
        # Step 3: Statistical
        print("\nStep 3: Running statistical analysis...")
        stats = StatisticalModels()
        statistical_results = stats.run_full_statistical_analysis(
            market_data=market_data,
            pricing_results=pricing_results,
            lookback_days=252
        )
        
        # Step 4: Risk
        risk_results = None
        if run_risk_analysis:
            print("\nStep 4: Running risk analysis...")
            risk = RiskAnalytics()
            risk_results = risk.run_full_risk_analysis(
                market_data=market_data,
                position_value=position_value,
                confidence_level=0.95,
                risk_free_rate=0.045,
                n_simulations=10000,
                time_horizon=252,
                run_scenarios=True
            )
        
        # Print summary
        print_summary(market_data, pricing_results, statistical_results, risk_results)
        
        # Save results
        if save_output:
            print("Saving results...")
            to_save = {
                'current_analysis': pricing_results['current_analysis'],
                'spread_stats': pricing_results['spread_stats'],
                'historical_basis': pricing_results['historical_basis'],
                'zscores': statistical_results['zscores'],
                'spread_predictions': statistical_results['spread_predictions']
            }
            
            if risk_results:
                to_save['risk_by_ticker'] = pd.DataFrame(risk_results['by_ticker']).T
                to_save['monte_carlo_by_ticker'] = pd.DataFrame(risk_results['monte_carlo']['by_ticker']).T
            
            save_results(to_save)
        
        print("Analysis complete\n")
        
        return {
            'market_data': market_data,
            'pricing_results': pricing_results,
            'statistical_results': statistical_results,
            'risk_results': risk_results,
            'status': 'SUCCESS'
        }
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'status': 'FAILED', 'error': str(e)}


if __name__ == "__main__":
    # Quick test: python main.py quick
    # Full analysis: python main.py
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        results = run_analysis(
            tickers=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2024-01-01',
            end_date='2024-12-31'
        )
    else:
        results = run_analysis()
    
    sys.exit(0 if results['status'] == 'SUCCESS' else 1)