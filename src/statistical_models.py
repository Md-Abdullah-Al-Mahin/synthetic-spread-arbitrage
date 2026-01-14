import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from pandas import Series
from sklearn.linear_model import LinearRegression

class StatisticalModels:
    """Statistical models for analyzing and forecasting synthetic spreads"""

    def __init__(self, lookback_days: int = 252):
        """
        Initialize statistical models
        """
        self.lookback_days = lookback_days

    def regression_spread_drivers(self,
                                  spreads: pd.DataFrame,
                                  volatility: pd.DataFrame,
                                  vix: Optional[pd.Series] = None,
                                  liquidity: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Multiple regression to understand what drives spread changes

        Formula: Spread = β₀ + β₁·Volatility + β₂·Liquidity + β₃·VIX + ε
        """
        print("Running regression analysis for spread drivers")

        results = {}
        tickers = list(set(spreads.columns) & set(volatility.columns))

        if not tickers:
            print("No common tickers between spreads and volatility data")
            return {}

        for ticker in tickers:
            try:
                print(f"  Analyzing {ticker}...", end="")

                # Prepare data for this ticker
                ticker_data = pd.DataFrame({
                    'spread': spreads[ticker].dropna(),
                    'volatility': volatility[ticker].dropna()
                })

                # Add VIX if provided
                if vix is not None:
                    ticker_data['vix'] = vix.reindex(ticker_data.index).fillna(method='ffill')

                # Add liquidity if provided
                if liquidity is not None and ticker in liquidity.columns:
                    ticker_data['liquidity'] = liquidity[ticker].reindex(ticker_data.index).fillna(method='ffill')

                # Drop any remaining NaN values
                ticker_data = ticker_data.dropna()

                if len(ticker_data) < 20:
                    print(f" skipped (insufficient data: {len(ticker_data)} obs)")
                    continue

                features = ticker_data.drop('spread', axis=1)
                target = ticker_data['spread']

                model = LinearRegression()
                model.fit(features, target)

                predictions = model.predict(features)
                residuals = target - predictions

                # R-squared
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((target - target.mean()) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                # Adjusted R-squared
                n = len(target)
                p = features.shape[1]
                adj_r_squared = 1 - (1 - r_squared) * ((n - 1) / (n - p - 1)) if n > p + 1 else r_squared

                # Store results
                ticker_results = {
                    'model': model,
                    'r_squared': r_squared,
                    'adj_r_squared': adj_r_squared,
                    'n_observations': n,
                    'coefficients': dict(zip(features.columns, model.coef_)),
                    'intercept': model.intercept_,
                    'feature_names': features.columns.tolist(),
                    'residuals': residuals
                }

                # Make a prediction example
                if len(features) > 0:
                    current_values = features.iloc[-1:].mean()
                    prediction = model.predict([current_values])[0]
                    ticker_results['current_prediction'] = prediction
                    ticker_results['current_actual'] = target.iloc[-1]
                    ticker_results['prediction_error'] = target.iloc[-1] - prediction

                results[ticker] = ticker_results
                print(f" done (R²={r_squared:.3f})")

            except Exception as e:
                print(f" error: {str(e)[:50]}")
                continue

        # Print summary
        if results:
            print(f"\nRegression Summary ({len(results)} tickers):")
            print(f"{'Ticker':<8} {'R²':<8} {'Adj R²':<8} {'Obs':<6} {'Pred Error':<12}")
            print("-" * 50)

            for ticker, res in results.items():
                error = res.get('prediction_error', 0)
                print(f"{ticker:<8} {res['r_squared']:.3f}    {res['adj_r_squared']:.3f}    "
                      f"{res['n_observations']:<6} {error:.6f}")

        return results

    # file name: src/statistical_models.py
    # Replace the existing calculate_regression_predictions with this:

    def calculate_regression_predictions(self,
                                         regression_results: Dict[str, Any],
                                         volatility: pd.DataFrame,
                                         vix: Optional[pd.Series] = None,
                                         liquidity: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate spread predictions using trained regression models
        """
        if volatility.empty:
            return pd.DataFrame()

        # Prepare empty predictions DataFrame with same index as volatility
        predictions = pd.DataFrame(index=volatility.index)

        for ticker, results in regression_results.items():
            if ticker not in volatility.columns:
                continue

            try:
                model = results['model']
                feature_names = results['feature_names']

                # Build feature matrix for this ticker
                feature_matrix = []
                valid_dates = []

                for date in volatility.index:
                    features = []
                    valid = True

                    for feat_name in feature_names:
                        if feat_name == 'volatility':
                            if date in volatility.index and ticker in volatility.columns:
                                features.append(volatility.loc[date, ticker])
                            else:
                                valid = False
                                break
                        elif feat_name == 'vix':
                            if vix is not None and date in vix.index:
                                features.append(vix.loc[date])
                            else:
                                valid = False
                                break
                        elif feat_name == 'liquidity':
                            if (liquidity is not None and
                                    ticker in liquidity.columns and
                                    date in liquidity.index):
                                features.append(liquidity.loc[date, ticker])
                            else:
                                valid = False
                                break
                        else:
                            valid = False
                            break

                    if valid:
                        feature_matrix.append(features)
                        valid_dates.append(date)

                if feature_matrix:
                    # Make predictions for all valid dates at once
                    ticker_predictions = model.predict(feature_matrix)
                    # Create Series for this ticker
                    pred_series = pd.Series(ticker_predictions, index=valid_dates, name=ticker)
                    # Merge into predictions DataFrame
                    predictions[ticker] = pred_series

            except Exception as e:
                print(f"Error predicting for {ticker}: {e}")
                continue

        return predictions