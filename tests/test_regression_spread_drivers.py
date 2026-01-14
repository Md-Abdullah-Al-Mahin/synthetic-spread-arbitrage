import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import DataPipeline
from src.statistical_models import StatisticalModels
from src.synthetic_pricer import SyntheticPricer


def generate_synthetic_spreads(pricer, prices, volatility):
    """Generate synthetic spread data using the SyntheticPricer"""
    spreads = pd.DataFrame(index=volatility.index)

    for ticker in volatility.columns:
        if ticker in prices.columns:
            # Calculate spreads for each day
            daily_spreads = []
            for date in volatility.index:
                try:
                    vol = volatility.loc[date, ticker]
                    spread = pricer.estimate_spread_from_volatility(vol)
                    daily_spreads.append(spread)
                except:
                    daily_spreads.append(np.nan)

            spreads[ticker] = daily_spreads

    return spreads.dropna(how='all')


def test_regression_with_real_vix():
    """Test regression with real VIX data from DataPipeline"""
    print("Testing Section 3.1 with Real VIX Data")
    print("=" * 60)

    try:
        # Initialize pipeline
        print("Initializing DataPipeline...")
        pipeline = DataPipeline()

        # Use smaller date range for quick testing
        pipeline.tickers = ['AAPL', 'MSFT', 'GOOGL'][:2]  # Just 2 for quick test
        start_date = '2024-01-01'
        end_date = '2024-06-01'

        print(f"Fetching data for {len(pipeline.tickers)} tickers")
        print(f"Date range: {start_date} to {end_date}")

        # Get complete market data including VIX
        data_dict = pipeline.get_market_data_complete(
            tickers=pipeline.tickers,
            start_date=start_date,
            end_date=end_date,
            include_vix=True,
            include_liquidity=False,
            force_download=False  # Set to True to force re-download
        )

        prices = data_dict['prices']
        volatility = data_dict['volatility']
        vix = data_dict['vix']

        print(f"\nData loaded:")
        print(f"  Prices: {prices.shape}")
        print(f"  Volatility: {volatility.shape}")
        print(f"  VIX: {len(vix) if vix is not None else 0} points")

        # Generate synthetic spreads
        print("\nGenerating synthetic spreads...")
        pricer = SyntheticPricer()
        spreads = generate_synthetic_spreads(pricer, prices, volatility)

        print(f"Synthetic spreads: {spreads.shape}")

        # Initialize statistical model
        model = StatisticalModels()

        # Test 1: Basic regression (volatility only)
        print("\n" + "=" * 60)
        print("Test 1: Regression with Volatility Only")
        print("=" * 60)

        results1 = model.regression_spread_drivers(
            spreads=spreads,
            volatility=volatility
        )

        # Test 2: Regression with VIX
        print("\n" + "=" * 60)
        print("Test 2: Regression with Volatility + VIX")
        print("=" * 60)

        if vix is not None and len(vix) > 0:
            results2 = model.regression_spread_drivers(
                spreads=spreads,
                volatility=volatility,
                vix=vix
            )
        else:
            print("Warning: No VIX data available")
            results2 = {}

        # Compare results
        print("\n" + "=" * 60)
        print("Comparison of Regression Results")
        print("=" * 60)

        for ticker in spreads.columns:
            if ticker in results1:
                res1 = results1[ticker]
                print(f"\n{ticker}:")
                print(f"  Volatility-only R²: {res1['r_squared']:.4f}")

                if ticker in results2:
                    res2 = results2[ticker]
                    print(f"  With VIX R²: {res2['r_squared']:.4f}")
                    improvement = res2['r_squared'] - res1['r_squared']
                    print(f"  Improvement: {improvement:.4f}")

                    # Show coefficients
                    if 'vix' in res2['coefficients']:
                        vix_coef = res2['coefficients']['vix']
                        print(f"  VIX coefficient: {vix_coef:.6f}")

        # Test predictions
        print("\n" + "=" * 60)
        print("Testing Predictions")
        print("=" * 60)

        if results2:
            predictions = model.calculate_regression_predictions(
                regression_results=results2,
                volatility=volatility,
                vix=vix
            )

            print(f"\nPredicted vs Actual Spreads (last day):")
            for ticker, pred in predictions.items():
                if ticker in spreads.columns:
                    actual = spreads[ticker].iloc[-1]
                    error = actual - pred
                    error_pct = (error / actual * 100) if actual != 0 else 0

                    print(f"  {ticker}: Pred={pred:.6f}, Actual={actual:.6f}, "
                          f"Error={error:.6f} ({error_pct:.1f}%)")

        return {
            'spreads': spreads,
            'volatility': volatility,
            'vix': vix,
            'results_vol_only': results1,
            'results_with_vix': results2
        }

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_vix_impact():
    """Analyze how VIX affects spreads"""
    print("\n" + "=" * 60)
    print("VIX Impact Analysis")
    print("=" * 60)

    # Create test data with different VIX regimes
    np.random.seed(42)

    # Simulate three regimes
    dates = pd.date_range('2024-01-01', '2024-06-01', freq='B')
    n = len(dates)

    # VIX regimes: low, medium, high
    vix_regimes = {
        'low': (dates[:n // 3], np.random.uniform(12, 18, n // 3)),
        'medium': (dates[n // 3:2 * n // 3], np.random.uniform(18, 25, n // 3)),
        'high': (dates[2 * n // 3:], np.random.uniform(25, 35, n // 3))
    }

    vix_series = pd.Series(index=dates, dtype=float)
    for regime_name, (regime_dates, regime_values) in vix_regimes.items():
        vix_series.loc[regime_dates] = regime_values

    # Generate spreads that respond to VIX
    base_spread = 0.0015
    vol_coefficient = 0.002
    vix_coefficient = 0.00005  # 0.5 bps per VIX point

    # Create volatility data
    volatility = {}
    spreads = {}

    for ticker in ['AAPL', 'MSFT']:
        # Base volatility with regime dependence
        base_vol = 0.2 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, n))
        # Add VIX correlation
        vol = base_vol + 0.001 * vix_series.values
        volatility[ticker] = np.clip(vol, 0.15, 0.4)

        # Generate spreads
        spread = (base_spread +
                  vol_coefficient * volatility[ticker] +
                  vix_coefficient * vix_series.values +
                  np.random.normal(0, 0.0002, n))
        spreads[ticker] = np.clip(spread, 0.001, 0.01)

    volatility_df = pd.DataFrame(volatility, index=dates)
    spreads_df = pd.DataFrame(spreads, index=dates)

    # Run regression
    model = StatisticalModels()

    print("\nRegression with simulated VIX regimes:")
    results = model.regression_spread_drivers(
        spreads=spreads_df,
        volatility=volatility_df,
        vix=vix_series
    )

    # Analyze by regime
    print("\nAnalysis by VIX Regime:")
    for ticker, res in results.items():
        vix_coef = res['coefficients'].get('vix', 0)
        print(f"\n{ticker}:")
        print(f"  VIX coefficient: {vix_coef:.6f}")
        print(f"  Interpretation: Each 1-point increase in VIX adds {vix_coef * 10000:.2f} bps to spread")

        # Calculate predicted spread change from low to high VIX
        vix_low = 15
        vix_high = 30
        vix_diff = vix_high - vix_low
        spread_change = vix_coef * vix_diff
        print(f"  Spread increase (VIX {vix_low}→{vix_high}): {spread_change * 10000:.2f} bps")