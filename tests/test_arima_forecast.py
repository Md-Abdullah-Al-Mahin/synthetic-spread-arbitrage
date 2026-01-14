import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from statistical_models import StatisticalModels

def create_trending_spread():
    """Create spread data with a clear trend"""
    np.random.seed(42)

    dates = pd.date_range('2024-01-01', periods=150, freq='B')

    # Create spread with upward trend
    trend = np.linspace(0.0015, 0.0030, len(dates))  # 15 to 30 bps
    noise = np.random.normal(0, 0.0002, len(dates))

    spread_values = trend + noise
    return pd.Series(spread_values, index=dates, name='TRENDING_SPREAD')


def create_mean_reverting_spread():
    """Create spread data that mean-reverts"""
    np.random.seed(43)

    dates = pd.date_range('2024-01-01', periods=150, freq='B')

    # Create mean-reverting spread
    spread_values = []
    current = 0.0020

    for i in range(len(dates)):
        # Mean reversion to 0.0020
        reversion = 0.1 * (0.0020 - current)
        noise = np.random.normal(0, 0.0003)
        current = current + reversion + noise
        spread_values.append(current)

    return pd.Series(spread_values, index=dates, name='MEAN_REVERTING_SPREAD')


def test_basic_arima():
    """Test basic ARIMA forecasting"""
    print("Test 1: Basic ARIMA Forecasting")
    print("=" * 50)

    # Create trending spread data
    spread_series = create_trending_spread()

    print(f"Spread data: {len(spread_series)} points")
    print(f"First value: {spread_series.iloc[0]:.6f}")
    print(f"Last value: {spread_series.iloc[-1]:.6f}")
    print(f"Trend: {((spread_series.iloc[-1] - spread_series.iloc[0]) / spread_series.iloc[0] * 100):.1f}%")

    # Initialize model
    model = StatisticalModels()

    # Fit ARIMA(1,1,1) - basic model
    print("\nFitting ARIMA(1,1,1) model...")
    result = model.fit_arima_model(
        series=spread_series,
        p=1,
        d=1,
        q=1,
        forecast_steps=5
    )

    if result:
        print(f"\nARIMA Results:")
        print(f"  Model: ARIMA{result['order']}")
        print(f"  AIC: {result['aic']:.2f}")
        print(f"  BIC: {result['bic']:.2f}")
        print(f"  Residual std: {result['residual_std']:.6f}")

        print(f"\nForecast:")
        for i, (date, value) in enumerate(result['forecast'].items()):
            lower = result['confidence_interval'].loc[date, 'lower']
            upper = result['confidence_interval'].loc[date, 'upper']
            print(f"  {date.date()}: {value:.6f} (95% CI: [{lower:.6f}, {upper:.6f}])")

        print(f"\nInterpretation:")
        print(f"  Current spread: {result['last_value']:.6f}")
        print(f"  5-day forecast: {result['forecast'].iloc[-1]:.6f}")
        print(f"  Forecast trend: {result['forecast_trend']}")

        # Trading implication
        current = result['last_value']
        forecast = result['forecast'].iloc[-1]
        if forecast > current:
            print(f"  → Forecast suggests spreads will WIDEN")
            print(f"  → Consider locking in financing rates now")
        else:
            print(f"  → Forecast suggests spreads will TIGHTEN")
            print(f"  → Consider waiting for better rates")

    return result


def test_multiple_tickers():
    """Test ARIMA forecasting for multiple tickers"""
    print("\n\nTest 2: Multiple Ticker Forecasting")
    print("=" * 50)

    dates = pd.date_range('2024-01-01', periods=120, freq='B')

    # Create DataFrame with different spread behaviors
    spreads_df = pd.DataFrame({
        'AAPL': create_trending_spread().values[:120],  # Trending up
        'MSFT': create_mean_reverting_spread().values[:120],  # Mean-reverting
        'GOOGL': 0.0025 + np.random.normal(0, 0.0003, 120)  # Random walk
    }, index=dates)

    # Initialize model
    model = StatisticalModels()

    # Forecast all tickers
    results = model.forecast_spread_trends(
        spreads=spreads_df,
        p=1,
        d=1,
        q=1,
        forecast_steps=5
    )

    if results:
        print(f"\nDetailed Forecasts:")
        for ticker, result in results.items():
            print(f"\n{ticker}:")
            print(f"  Current: {result['last_value']:.6f}")
            print(f"  5-day forecast: {result['forecast'].iloc[-1]:.6f}")
            print(f"  Change: {(result['forecast'].iloc[-1] - result['last_value']) * 10000:+.1f} bps")
            print(f"  Trend: {result['forecast_trend']}")

            # ARIMA parameters
            print(f"  Key parameters:")
            params = result['params']
            for key in ['ar.L1', 'ma.L1']:
                if key in params:
                    print(f"    {key}: {params[key]:.4f}")

    return results


def test_order_comparison():
    """Test comparing different ARIMA orders"""
    print("\n\nTest 3: ARIMA Order Comparison")
    print("=" * 50)

    # Use mean-reverting spread for this test
    spread_series = create_mean_reverting_spread()

    # Initialize model
    model = StatisticalModels()

    # Compare different orders
    comparison = model.compare_arima_orders(
        series=spread_series,
        forecast_steps=5
    )

    if not comparison.empty:
        print(f"\nUsing best model for forecasting...")
        best_row = comparison.iloc[0]

        best_result = model.fit_arima_model(
            series=spread_series,
            p=int(best_row['p']),
            d=int(best_row['d']),
            q=int(best_row['q']),
            forecast_steps=5
        )

        if best_result:
            print(f"\nBest Model Forecast:")
            print(f"  ARIMA{best_result['order']}")
            print(f"  Current: {best_result['last_value']:.6f}")
            print(f"  5-day forecast: {best_result['forecast'].iloc[-1]:.6f}")
            print(f"  Trend: {best_result['forecast_trend']}")

    return comparison


def test_integration_with_zscore():
    """Test integration with Z-score analysis"""
    print("\n\nTest 4: Integration with Z-Score Analysis")
    print("=" * 50)

    # Create spread data
    spread_series = create_mean_reverting_spread()

    # Initialize model
    model = StatisticalModels(lookback_days=60)

    print("Step 1: Calculate Z-score (Section 3.3)")
    zscores = model.calculate_zscore(spread_series)
    current_z = zscores.iloc[-1]
    z_interp = model.interpret_zscore(current_z, threshold=2.0)

    print(f"  Current Z-score: {current_z:.2f}")
    print(f"  Signal: {z_interp['signal']}")
    print(f"  Interpretation: {z_interp['interpretation']}")

    print("\nStep 2: ARIMA forecast (Section 3.4)")
    arima_result = model.fit_arima_model(
        series=spread_series,
        forecast_steps=5
    )

    if arima_result:
        current = arima_result['last_value']
        forecast = arima_result['forecast'].iloc[-1]
        change = forecast - current
        change_bps = change * 10000

        print(f"  Current spread: {current:.6f}")
        print(f"  5-day forecast: {forecast:.6f}")
        print(f"  Expected change: {change_bps:+.1f} bps")
        print(f"  Forecast trend: {arima_result['forecast_trend']}")

        print("\nCombined Analysis:")
        if z_interp['signal'] == 'LONG_BASIS' and change_bps > 0:
            print("  → Synthetic is CHEAP and expected to get MORE expensive")
            print("  → STRONG BUY signal: Enter synthetic position now")
        elif z_interp['signal'] == 'SHORT_BASIS' and change_bps < 0:
            print("  → Synthetic is EXPENSIVE and expected to get CHEAPER")
            print("  → STRONG AVOID signal: Use cash instead")
        elif z_interp['signal'] == 'NEUTRAL':
            print("  → Spread is within normal range")
            if change_bps > 1.0:
                print("  → But expecting WIDENING, consider locking rates")
            elif change_bps < -1.0:
                print("  → But expecting TIGHTENING, consider waiting")
            else:
                print("  → No clear directional signal")

    return {
        'zscore': current_z,
        'z_interp': z_interp,
        'arima_result': arima_result
    }