import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from statistical_models import StatisticalModels


def create_test_regression_data():
    """Create realistic test data for regression predictions"""
    np.random.seed(42)

    # Create date range
    dates = pd.date_range(start='2024-01-01', periods=100, freq='B')

    # Create tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    # Create volatility data
    volatility_data = {}
    for ticker in tickers:
        # Generate realistic volatility patterns
        base = 0.2 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, len(dates)))
        noise = np.random.normal(0, 0.03, len(dates))
        volatility_data[ticker] = np.clip(base + noise, 0.15, 0.5)

    volatility_df = pd.DataFrame(volatility_data, index=dates)

    # Create VIX data
    vix_base = 15 + 5 * np.sin(np.linspace(0, 2 * np.pi, len(dates)))
    vix_noise = np.random.normal(0, 1.5, len(dates))
    vix_series = pd.Series(
        np.clip(vix_base + vix_noise, 10, 35),
        index=dates,
        name='VIX'
    )

    # Create liquidity data
    liquidity_data = {}
    for ticker in tickers:
        base = 0.001 + 0.0003 * np.sin(np.linspace(0, np.pi, len(dates)))
        noise = np.random.normal(0, 0.0001, len(dates))
        liquidity_data[ticker] = np.clip(base + noise, 0.0005, 0.002)

    liquidity_df = pd.DataFrame(liquidity_data, index=dates)

    # Create spreads (target variable) using a known relationship
    spreads_data = {}
    for ticker in tickers:
        # Known coefficients for each ticker
        if ticker == 'AAPL':
            coefficients = {'intercept': 0.0015, 'volatility': 0.002, 'vix': 0.00004, 'liquidity': 0.3}
        elif ticker == 'MSFT':
            coefficients = {'intercept': 0.0012, 'volatility': 0.0018, 'vix': 0.00003, 'liquidity': 0.25}
        else:  # GOOGL
            coefficients = {'intercept': 0.0018, 'volatility': 0.0022, 'vix': 0.00005, 'liquidity': 0.35}

        # Calculate spread using the relationship
        spread = (coefficients['intercept'] +
                  coefficients['volatility'] * volatility_df[ticker] +
                  coefficients['vix'] * vix_series +
                  coefficients['liquidity'] * liquidity_df[ticker] +
                  np.random.normal(0, 0.0002, len(dates)))

        spreads_data[ticker] = np.clip(spread, 0.001, 0.01)

    spreads_df = pd.DataFrame(spreads_data, index=dates)

    return {
        'spreads': spreads_df,
        'volatility': volatility_df,
        'vix': vix_series,
        'liquidity': liquidity_df,
        'dates': dates
    }


def test_basic_predictions():
    """Test basic prediction functionality"""
    print("Test 1: Basic Predictions")
    print("=" * 50)

    # Create test data
    test_data = create_test_regression_data()

    # Initialize model
    model = StatisticalModels()

    # Run regression with all features
    results = model.regression_spread_drivers(
        spreads=test_data['spreads'],
        volatility=test_data['volatility'],
        vix=test_data['vix'],
        liquidity=test_data['liquidity']
    )

    # Generate predictions for all dates
    predictions = model.calculate_regression_predictions(
        regression_results=results,
        volatility=test_data['volatility'],
        vix=test_data['vix'],
        liquidity=test_data['liquidity']
    )

    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions columns: {list(predictions.columns)}")
    print(f"Predictions date range: {predictions.index[0].date()} to {predictions.index[-1].date()}")

    # Check that we got predictions for all tickers
    for ticker in test_data['spreads'].columns:
        if ticker in predictions.columns:
            print(f"\n{ticker}:")
            print(f"  Number of predictions: {predictions[ticker].count()}")
            print(f"  Prediction range: [{predictions[ticker].min():.6f}, {predictions[ticker].max():.6f}]")
        else:
            print(f"\n{ticker}: No predictions generated")

    return predictions, test_data, results


def test_prediction_accuracy(predictions=test_basic_predictions()[0], test_data=test_basic_predictions()[1]):
    """Test prediction accuracy against known values"""
    print("\n\nTest 2: Prediction Accuracy")
    print("=" * 50)

    spreads = test_data['spreads']

    # Calculate prediction errors
    errors = pd.DataFrame()

    for ticker in predictions.columns:
        if ticker in spreads.columns:
            # Align dates
            common_dates = predictions.index.intersection(spreads.index)
            if len(common_dates) > 0:
                pred_series = predictions.loc[common_dates, ticker]
                actual_series = spreads.loc[common_dates, ticker]

                # Calculate error metrics
                mae = np.mean(np.abs(pred_series - actual_series))
                rmse = np.sqrt(np.mean((pred_series - actual_series) ** 2))
                mape = np.mean(np.abs((actual_series - pred_series) / actual_series)) * 100

                errors.loc[ticker, 'MAE'] = mae
                errors.loc[ticker, 'RMSE'] = rmse
                errors.loc[ticker, 'MAPE'] = mape
                errors.loc[ticker, 'Correlation'] = pred_series.corr(actual_series)

    print("Error Metrics:")
    print(errors.round(6))

    # Check if predictions are reasonable
    print("\nAccuracy Check:")
    for ticker in errors.index:
        if errors.loc[ticker, 'MAE'] < 0.001:  # Less than 1 bp error on average
            print(f"  {ticker}: ✓ Good accuracy (MAE = {errors.loc[ticker, 'MAE']:.6f})")
        else:
            print(f"  {ticker}: ✗ High error (MAE = {errors.loc[ticker, 'MAE']:.6f})")

    return errors


def test_partial_data_predictions():
    """Test predictions with partial/missing data"""
    print("\n\nTest 3: Partial Data Predictions")
    print("=" * 50)

    test_data = create_test_regression_data()
    model = StatisticalModels()

    # Test 3a: Predictions without VIX
    print("\n3a: Predictions without VIX (volatility only)")
    results_vol_only = model.regression_spread_drivers(
        spreads=test_data['spreads'],
        volatility=test_data['volatility']
    )

    preds_vol_only = model.calculate_regression_predictions(
        regression_results=results_vol_only,
        volatility=test_data['volatility']
    )

    print(f"  Predictions shape: {preds_vol_only.shape}")
    print(f"  Missing VIX handled: {'Yes' if not preds_vol_only.empty else 'No'}")

    # Test 3b: Predictions with missing liquidity data
    print("\n3b: Predictions with VIX but no liquidity")
    results_with_vix = model.regression_spread_drivers(
        spreads=test_data['spreads'],
        volatility=test_data['volatility'],
        vix=test_data['vix']
    )

    preds_with_vix = model.calculate_regression_predictions(
        regression_results=results_with_vix,
        volatility=test_data['volatility'],
        vix=test_data['vix']
    )

    print(f"  Predictions shape: {preds_with_vix.shape}")

    # Test 3c: Predictions with different date range
    print("\n3c: Predictions for subset of dates")
    subset_dates = test_data['dates'][-20:]  # Last 20 days
    subset_volatility = test_data['volatility'].loc[subset_dates]
    subset_vix = test_data['vix'].loc[subset_dates]

    preds_subset = model.calculate_regression_predictions(
        regression_results=results_with_vix,
        volatility=subset_volatility,
        vix=subset_vix
    )

    print(f"  Subset predictions shape: {preds_subset.shape}")
    print(f"  Subset date range: {preds_subset.index[0].date()} to {preds_subset.index[-1].date()}")

    return {
        'vol_only': preds_vol_only,
        'with_vix': preds_with_vix,
        'subset': preds_subset
    }


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n\nTest 4: Edge Cases")
    print("=" * 50)

    model = StatisticalModels()

    # Test 4a: Empty data
    print("\n4a: Empty data")
    empty_results = {}
    empty_vol = pd.DataFrame()

    try:
        preds_empty = model.calculate_regression_predictions(
            regression_results=empty_results,
            volatility=empty_vol
        )
        print(f"  Empty input handled: Yes, shape = {preds_empty.shape}")
    except Exception as e:
        print(f"  Empty input error: {e}")

    # Test 4b: Missing ticker in volatility
    print("\n4b: Missing ticker in volatility")
    test_data = create_test_regression_data()

    # Create results for a ticker not in volatility
    fake_results = {
        'FAKE': {
            'model': None,
            'feature_names': ['volatility']
        }
    }

    try:
        preds_missing = model.calculate_regression_predictions(
            regression_results=fake_results,
            volatility=test_data['volatility']
        )
        print(f"  Missing ticker handled: Yes, shape = {preds_missing.shape}")
    except Exception as e:
        print(f"  Missing ticker error: {e}")

    # Test 4c: Single date prediction
    print("\n4c: Single date prediction")
    results = model.regression_spread_drivers(
        spreads=test_data['spreads'],
        volatility=test_data['volatility']
    )

    # Use only the last date
    last_date = test_data['dates'][-1]
    single_vol = test_data['volatility'].loc[[last_date]]

    preds_single = model.calculate_regression_predictions(
        regression_results=results,
        volatility=single_vol
    )

    print(f"  Single date shape: {preds_single.shape}")
    print(f"  Single date predictions: {preds_single.iloc[0].round(6).to_dict()}")


def integration_test_with_synthetic_pricer():
    """Integration test using SyntheticPricer to generate spreads"""
    print("\n\nTest 5: Integration with SyntheticPricer")
    print("=" * 50)

    try:
        # Import SyntheticPricer
        from synthetic_pricer import SyntheticPricer

        # Create test data
        test_data = create_test_regression_data()

        # Use SyntheticPricer to generate spreads
        pricer = SyntheticPricer()

        # Generate spreads using the pricer's volatility-spread relationship
        spreads_from_pricer = pd.DataFrame(index=test_data['dates'])

        for ticker in test_data['volatility'].columns:
            spreads_from_pricer[ticker] = test_data['volatility'][ticker].apply(
                lambda vol: pricer.estimate_spread_from_volatility(vol)
            )

        # Initialize statistical model
        model = StatisticalModels()

        # Run regression
        print("Running regression with SyntheticPricer spreads...")
        results = model.regression_spread_drivers(
            spreads=spreads_from_pricer,
            volatility=test_data['volatility']
        )

        # Generate predictions
        predictions = model.calculate_regression_predictions(
            regression_results=results,
            volatility=test_data['volatility']
        )

        # Compare with actual spreads from pricer
        print("\nComparison (last 5 days):")
        comparison = pd.DataFrame({
            'Actual': spreads_from_pricer.iloc[-5:, 0],
            'Predicted': predictions.iloc[-5:, 0]
        })
        print(comparison.round(6))

        # Calculate error
        mae = np.mean(np.abs(predictions.iloc[:, 0] - spreads_from_pricer.iloc[:, 0]))
        print(f"\nMean Absolute Error: {mae:.6f}")
        print(f"Model correctly captured volatility-spread relationship: {'Yes' if mae < 0.0005 else 'No'}")

        return predictions, spreads_from_pricer

    except ImportError as e:
        print(f"SyntheticPricer not available: {e}")
        return None, None
