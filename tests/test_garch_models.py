# file name: test_garch_models.py
"""
Test script for Section 3.2: GARCH Volatility Modeling
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from statistical_models import StatisticalModels


def create_test_returns_data():
    """Create realistic returns data for GARCH testing"""
    np.random.seed(42)

    # Create date range (need enough data for GARCH)
    dates = pd.date_range(start='2022-01-01', periods=500, freq='B')

    # Create tickers
    tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA']

    # Create returns with different volatility characteristics
    returns_data = {}

    # AAPL: Moderate persistence, moderate volatility
    print("Generating AAPL returns with moderate persistence...")
    returns_data['AAPL'] = generate_garch_returns(
        n=500, omega=0.01, alpha=0.1, beta=0.85, mu=0.0005, sigma=0.01
    )

    # MSFT: High persistence (volatility clusters)
    print("Generating MSFT returns with high persistence...")
    returns_data['MSFT'] = generate_garch_returns(
        n=500, omega=0.005, alpha=0.15, beta=0.8, mu=0.0003, sigma=0.008
    )

    # TSLA: High volatility, low persistence (mean-reverting quickly)
    print("Generating TSLA returns with low persistence...")
    returns_data['TSLA'] = generate_garch_returns(
        n=500, omega=0.02, alpha=0.05, beta=0.6, mu=0.001, sigma=0.02
    )

    # NVDA: Very high persistence (long volatility memory)
    print("Generating NVDA returns with very high persistence...")
    returns_data['NVDA'] = generate_garch_returns(
        n=500, omega=0.001, alpha=0.1, beta=0.88, mu=0.0008, sigma=0.015
    )

    returns_df = pd.DataFrame(returns_data, index=dates)
    print(f"Created returns data: {returns_df.shape}")

    return returns_df


def generate_garch_returns(n=500, omega=0.01, alpha=0.1, beta=0.85, mu=0.0005, sigma=0.01):
    """Generate returns using GARCH(1,1) process"""
    # Initialize arrays
    returns = np.zeros(n)
    variance = np.zeros(n)

    # Initial variance
    variance[0] = omega / (1 - alpha - beta)

    # Generate GARCH process
    for t in range(1, n):
        # Generate return with time-varying volatility
        returns[t] = mu + np.sqrt(variance[t - 1]) * np.random.normal(0, 1)

        # Update variance
        variance[t] = omega + alpha * (returns[t - 1] - mu) ** 2 + beta * variance[t - 1]

    return returns


def test_garch_fitting():
    """Test basic GARCH model fitting"""
    print("\n" + "=" * 60)
    print("Test 1: GARCH Model Fitting")
    print("=" * 60)

    # Create test returns
    returns = create_test_returns_data()

    # Initialize model
    model = StatisticalModels()

    # Fit GARCH models
    print("\nFitting GARCH(1,1) models...")
    garch_results = model.fit_garch_volatility(
        returns=returns,
        tickers=['AAPL', 'MSFT', 'TSLA', 'NVDA'],
        p=1,
        q=1,
        forecast_horizon=10
    )

    # Analyze persistence
    print("\nAnalyzing GARCH persistence...")
    persistence_df = model.analyze_garch_persistence(garch_results)

    # Display detailed results
    print("\n" + "=" * 60)
    print("Detailed GARCH Results")
    print("=" * 60)

    for ticker, results in garch_results.items():
        print(f"\n{ticker}:")
        params = results['params']
        print(f"  Parameters:")
        print(f"    ω (omega): {params['omega']:.6f}")
        print(f"    α (alpha): {params['alpha']:.4f}")
        print(f"    β (beta): {params['beta']:.4f}")
        print(f"    Persistence (α+β): {params['persistence']:.4f}")
        print(f"    Long-run volatility: {params['long_run_vol']:.3%}")

        print(f"  Current volatility: {results['current_volatility']:.3%}")
        print(f"  Expected vol change (10-day): {results['expected_vol_change']:.3%}")

        print(f"  Model fit:")
        print(f"    Log-likelihood: {results['log_likelihood']:.2f}")
        print(f"    AIC: {results['aic']:.2f}")
        print(f"    BIC: {results['bic']:.2f}")

        print(f"  10-day volatility forecast:")
        for i, vol in enumerate(results['forecast_volatility']):
            print(f"    Day {i + 1}: {vol:.3%}")

    return garch_results, returns


def test_volatility_forecasting(garch_results=test_garch_fitting()[0], returns=test_garch_fitting()[1]):
    """Test volatility forecasting functionality"""
    print("\n" + "=" * 60)
    print("Test 2: Volatility Forecasting")
    print("=" * 60)

    model = StatisticalModels()

    # Generate forecasts
    volatility_forecasts = model.forecast_volatility_using_garch(
        returns=returns,
        garch_results=garch_results,
        forecast_horizon=10
    )

    print(f"\nVolatility Forecasts DataFrame:")
    print(f"Shape: {volatility_forecasts.shape}")
    print(f"Forecast dates: {volatility_forecasts.index[0].date()} to {volatility_forecasts.index[-1].date()}")

    # Display forecasts
    print("\nFirst 5 days of volatility forecasts:")
    print(volatility_forecasts.head().round(4))

    # Calculate forecast statistics
    print("\nForecast Statistics:")
    for ticker in volatility_forecasts.columns:
        forecast_series = volatility_forecasts[ticker]
        start_vol = forecast_series.iloc[0]
        end_vol = forecast_series.iloc[-1]
        change_pct = (end_vol - start_vol) / start_vol * 100

        print(f"  {ticker}: Start={start_vol:.3%}, End={end_vol:.3%}, "
              f"Change={change_pct:.1f}%")

    return volatility_forecasts


def test_integration_with_regression():
    """Test integration of GARCH with regression models"""
    print("\n" + "=" * 60)
    print("Test 3: GARCH + Regression Integration")
    print("=" * 60)

    # Create test data
    returns = create_test_returns_data()

    # Calculate volatility from returns (for regression)
    volatility = returns.rolling(30).std() * np.sqrt(252)
    volatility = volatility.dropna()

    # Create synthetic spreads (for regression)
    spreads = pd.DataFrame(index=volatility.index)
    for ticker in volatility.columns:
        # Simple relationship: spread = 0.0015 + 0.002 * volatility
        spreads[ticker] = 0.0015 + 0.002 * volatility[ticker] + np.random.normal(0, 0.0001, len(volatility))

    # Initialize model
    model = StatisticalModels()

    # Step 1: Fit regression models (from Section 3.1)
    print("\n1. Fitting regression models...")
    regression_results = model.regression_spread_drivers(
        spreads=spreads,
        volatility=volatility
    )

    # Step 2: Fit GARCH models
    print("\n2. Fitting GARCH models...")
    garch_results = model.fit_garch_volatility(
        returns=returns.loc[volatility.index],  # Align with volatility dates
        forecast_horizon=10
    )

    # Step 3: Forecast volatility
    print("\n3. Forecasting volatility...")
    volatility_forecasts = model.forecast_volatility_using_garch(
        returns=returns,
        garch_results=garch_results,
        forecast_horizon=10
    )

    # Step 4: Forecast spreads
    print("\n4. Forecasting spreads...")
    current_spreads = spreads.loc[spreads.index[-30:]]  # Recent spreads

    spread_forecasts = model.calculate_garch_based_spread_forecast(
        garch_results=garch_results,
        regression_results=regression_results,
        current_spreads=current_spreads,
        volatility_forecasts=volatility_forecasts
    )

    # Display results
    if not spread_forecasts.empty:
        print(f"\nSpread Forecasts:")
        print(f"Shape: {spread_forecasts.shape}")

        print("\nFirst 5 days of spread forecasts:")
        print(spread_forecasts.head().round(6))

        # Calculate expected financing cost changes
        print("\nExpected Financing Cost Impact:")
        for ticker in spread_forecasts.columns:
            if ticker in current_spreads.columns:
                current = current_spreads[ticker].iloc[-1]
                forecast = spread_forecasts[ticker].iloc[-1]
                change = forecast - current
                change_bps = change * 10000  # Convert to basis points

                print(f"  {ticker}: {current:.6f} → {forecast:.6f}, "
                      f"Δ={change:.6f} ({change_bps:.1f} bps)")

    return {
        'regression_results': regression_results,
        'garch_results': garch_results,
        'volatility_forecasts': volatility_forecasts,
        'spread_forecasts': spread_forecasts
    }


def test_garch_diagnostic_plots():
    """Test GARCH model diagnostics"""
    print("\n" + "=" * 60)
    print("Test 4: GARCH Model Diagnostics")
    print("=" * 60)

    # Create test returns
    returns = create_test_returns_data()

    # Initialize model
    model = StatisticalModels()

    # Fit GARCH for one ticker
    print("\nFitting GARCH model for AAPL...")
    garch_results = model.fit_garch_volatility(
        returns=returns[['AAPL']],
        forecast_horizon=10
    )

    if 'AAPL' in garch_results:
        results = garch_results['AAPL']
        fitted = results['fitted_model']

        print("\nModel Diagnostics:")
        print(fitted.summary())

        # Check model assumptions
        print("\nChecking Model Assumptions:")

        # 1. Residuals should have no autocorrelation
        residuals = fitted.resid
        print(f"  Residual mean: {residuals.mean():.6f}")
        print(f"  Residual std: {residuals.std():.6f}")

        # 2. Standardized residuals should be normal
        standardized_residuals = residuals / np.sqrt(fitted.conditional_volatility)
        from scipy import stats
        jb_stat, jb_pvalue = stats.jarque_bera(standardized_residuals.dropna())
        print(f"  Jarque-Bera test for normality:")
        print(f"    Statistic: {jb_stat:.2f}, p-value: {jb_pvalue:.4f}")
        print(f"    Normal distribution: {'Yes' if jb_pvalue > 0.05 else 'No'}")

        # 3. Check for ARCH effects in residuals
        # (If GARCH is correctly specified, there should be no ARCH effects)
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(standardized_residuals.dropna() ** 2, lags=[10], return_df=True)
        print(f"  Ljung-Box test for ARCH effects:")
        print(f"    Q-statistic: {lb_test['lb_stat'].iloc[0]:.2f}, p-value: {lb_test['lb_pvalue'].iloc[0]:.4f}")
        print(f"    No ARCH effects: {'Yes' if lb_test['lb_pvalue'].iloc[0] > 0.05 else 'No'}")


def test_different_garch_specifications():
    """Test different GARCH model specifications"""
    print("\n" + "=" * 60)
    print("Test 5: Different GARCH Specifications")
    print("=" * 60)

    # Create test returns
    returns = create_test_returns_data()

    # Initialize model
    model = StatisticalModels()

    # Test different GARCH specifications
    specifications = [
        (1, 1, 'GARCH(1,1)'),
        (2, 1, 'GARCH(2,1)'),
        (1, 2, 'GARCH(1,2)'),
        (2, 2, 'GARCH(2,2)')
    ]

    comparison_results = []

    for p, q, name in specifications:
        print(f"\nTesting {name}...")

        try:
            garch_results = model.fit_garch_volatility(
                returns=returns[['AAPL']],
                p=p,
                q=q,
                forecast_horizon=10
            )

            if 'AAPL' in garch_results:
                results = garch_results['AAPL']

                comparison_results.append({
                    'Model': name,
                    'p': p,
                    'q': q,
                    'LogLikelihood': results['log_likelihood'],
                    'AIC': results['aic'],
                    'BIC': results['bic'],
                    'Persistence': results['params']['persistence'],
                    'Current_Vol': results['current_volatility'],
                    'Forecast_Vol': results['forecast_volatility'][-1]
                })

                print(f"  Log-likelihood: {results['log_likelihood']:.2f}")
                print(f"  AIC: {results['aic']:.2f}")
                print(f"  Persistence: {results['params']['persistence']:.4f}")

        except Exception as e:
            print(f"  Error with {name}: {e}")
            continue

    # Compare specifications
    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)

        print("\n" + "=" * 60)
        print("Model Comparison:")
        print(comparison_df.to_string(index=False))

        # Find best model by AIC (lower is better)
        best_by_aic = comparison_df.loc[comparison_df['AIC'].idxmin()]
        print(f"\nBest model by AIC: {best_by_aic['Model']} (AIC={best_by_aic['AIC']:.2f})")

        # Find best model by BIC (lower is better)
        best_by_bic = comparison_df.loc[comparison_df['BIC'].idxmin()]
        print(f"Best model by BIC: {best_by_bic['Model']} (BIC={best_by_bic['BIC']:.2f})")