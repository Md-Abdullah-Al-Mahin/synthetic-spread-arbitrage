import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pandas import Series
from sklearn.linear_model import LinearRegression
from arch import  arch_model

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

    def fit_garch_volatility(self,
                             returns: pd.DataFrame,
                             tickers: Optional[List[str]] = None,
                             p: int = 1,
                             q: int = 1,
                             forecast_horizon: int = 10) -> Dict[str, Any]:
        """
        Fit GARCH(1,1) model and forecast future volatility

        Formula: σ²ₜ = ω + α·r²ₜ₋₁ + β·σ²ₜ₋₁
        """
        print(f"Fitting GARCH({p},{q}) models...")

        if tickers is None:
            tickers = returns.columns.tolist()

        results = {}

        for ticker in tickers:
            if ticker not in returns.columns:
                print(f"  {ticker}: Not found in returns data")
                continue

            print(f"  {ticker}...", end="")

            try:
                # Get returns for this ticker
                ticker_returns = returns[ticker].dropna()

                if len(ticker_returns) < 100:
                    print(f" skipped (insufficient data: {len(ticker_returns)} obs)")
                    continue

                # Convert to percentage (GARCH models often work better with percentages)
                returns_pct = ticker_returns * 100

                # Fit GARCH model
                model = arch_model(
                    returns_pct,
                    vol='GARCH',
                    p=p,
                    q=q,
                    dist='normal'
                )

                fitted = model.fit(disp='off', show_warning=False)

                # Get model parameters
                params = fitted.params
                omega = params.get('omega', 0)
                alpha = params.get('alpha[1]', 0) if p >= 1 else 0
                beta = params.get('beta[1]', 0) if q >= 1 else 0

                # Calculate persistence (alpha + beta)
                persistence = alpha + beta

                # Calculate long-run variance and volatility
                if persistence < 1:
                    long_run_variance = omega / (1 - persistence)
                    long_run_vol = np.sqrt(long_run_variance)
                else:
                    long_run_variance = np.nan
                    long_run_vol = np.nan

                # Forecast volatility
                forecast = fitted.forecast(horizon=forecast_horizon)
                forecast_variance = forecast.variance.iloc[-1]  # Last row has forecasts
                forecast_vol = np.sqrt(forecast_variance) / 100  # Convert back to decimal

                # Get current volatility
                current_vol = np.sqrt(fitted.conditional_volatility.iloc[-1]) / 100

                # Calculate expected spread change based on forecast
                expected_vol_change = forecast_vol.iloc[-1] - current_vol

                # Store results
                ticker_results = {
                    'fitted_model': fitted,
                    'params': {
                        'omega': omega,
                        'alpha': alpha,
                        'beta': beta,
                        'persistence': persistence,
                        'long_run_variance': long_run_variance,
                        'long_run_vol': long_run_vol
                    },
                    'current_volatility': current_vol,
                    'forecast_horizon': forecast_horizon,
                    'forecast_volatility': forecast_vol.tolist(),
                    'forecast_variance': forecast_variance.tolist(),
                    'expected_vol_change': expected_vol_change,
                    'volatility_forecast_date': returns.index[-1] + pd.Timedelta(days=1),
                    'n_observations': len(ticker_returns),
                    'log_likelihood': fitted.loglikelihood,
                    'aic': fitted.aic,
                    'bic': fitted.bic
                }

                results[ticker] = ticker_results
                print(f" done (persistence: {persistence:.3f})")

            except Exception as e:
                print(f" error: {str(e)[:50]}")
                continue

        # Print summary
        if results:
            print(f"\nGARCH Model Summary ({len(results)} tickers):")
            print(f"{'Ticker':<8} {'Current Vol':<12} {'Forecast Vol':<12} {'α':<8} {'β':<8} {'Persistence':<12}")
            print("-" * 60)

            for ticker, res in results.items():
                params = res['params']
                current = res['current_volatility']
                forecast = res['forecast_volatility'][-1] if len(res['forecast_volatility']) > 0 else np.nan

                print(f"{ticker:<8} {current:.3%}        {forecast:.3%}        "
                      f"{params['alpha']:.4f}   {params['beta']:.4f}   {params['persistence']:.4f}")

        return results

    def forecast_volatility_using_garch(self,
                                        returns: pd.DataFrame,
                                        garch_results: Dict[str, Any],
                                        forecast_horizon: int = 10) -> pd.DataFrame:
        """
        Generate volatility forecasts using pre-trained GARCH models
        """
        if not garch_results:
            return pd.DataFrame()

        # Create future dates for forecasting
        last_date = returns.index[-1]
        future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(forecast_horizon)]

        forecasts = pd.DataFrame(index=future_dates)

        for ticker, results in garch_results.items():
            try:
                # Get the fitted model
                fitted_model = results['fitted_model']

                # Generate forecast
                forecast = fitted_model.forecast(horizon=forecast_horizon)
                forecast_vol = np.sqrt(forecast.variance.iloc[-1]) / 100  # Convert to decimal

                # Store in DataFrame
                forecasts[ticker] = forecast_vol.values

            except Exception as e:
                print(f"Error forecasting for {ticker}: {e}")
                continue

        return forecasts

    def analyze_garch_persistence(self,
                                  garch_results: Dict[str, Any],
                                  persistence_threshold: float = 0.95) -> pd.DataFrame:
        """
        Analyze GARCH persistence across tickers
        """
        analysis_data = []

        for ticker, results in garch_results.items():
            params = results['params']
            persistence = params['persistence']

            # Classify persistence
            if persistence >= persistence_threshold:
                persistence_class = 'HIGH'
            elif persistence >= 0.85:
                persistence_class = 'MEDIUM'
            else:
                persistence_class = 'LOW'

            # Calculate half-life of volatility shocks
            if persistence > 0:
                half_life = np.log(0.5) / np.log(persistence)
            else:
                half_life = np.inf

            analysis_data.append({
                'Ticker': ticker,
                'Persistence': persistence,
                'Persistence_Class': persistence_class,
                'Alpha': params['alpha'],
                'Beta': params['beta'],
                'Omega': params['omega'],
                'Long_Run_Vol': params['long_run_vol'],
                'Half_Life_Days': half_life,
                'Current_Vol': results['current_volatility'],
                'Expected_Vol_Change': results['expected_vol_change']
            })

        if not analysis_data:
            return pd.DataFrame()

        analysis_df = pd.DataFrame(analysis_data).set_index('Ticker')

        # Sort by persistence
        analysis_df = analysis_df.sort_values('Persistence', ascending=False)

        print(f"\nGARCH Persistence Analysis:")
        print(f"{'Ticker':<8} {'Persistence':<12} {'Class':<8} {'Half-Life':<12} {'Expected ΔVol':<15}")
        print("-" * 60)

        for ticker, row in analysis_df.iterrows():
            print(f"{ticker:<8} {row['Persistence']:.4f}      "
                  f"{row['Persistence_Class']:<8} {row['Half_Life_Days']:.1f} days   "
                  f"{row['Expected_Vol_Change']:.3%}")

        return analysis_df

    def calculate_garch_based_spread_forecast(self,
                                              garch_results: Dict[str, Any],
                                              regression_results: Dict[str, Any],
                                              current_spreads: pd.DataFrame,
                                              volatility_forecasts: pd.DataFrame) -> pd.DataFrame:
        """
        Combine GARCH volatility forecasts with regression models to forecast spreads
        """
        if volatility_forecasts.empty:
            return pd.DataFrame()

        spread_forecasts = pd.DataFrame(index=volatility_forecasts.index)

        for ticker in volatility_forecasts.columns:
            if (ticker in regression_results and
                    ticker in current_spreads.columns and
                    ticker in garch_results):

                try:
                    # Get regression model
                    reg_model = regression_results[ticker]['model']
                    feature_names = regression_results[ticker]['feature_names']

                    # Get current spread
                    current_spread = current_spreads[ticker].iloc[-1] if len(current_spreads[ticker]) > 0 else np.nan

                    # Forecast spreads for each future date
                    forecast_spreads = []

                    for date in volatility_forecasts.index:
                        # Prepare features
                        features = []

                        # Volatility is always a feature
                        vol_feature = volatility_forecasts.loc[date, ticker]

                        # For now, we only use volatility forecast
                        # In a full implementation, we'd also forecast VIX and liquidity
                        if feature_names == ['volatility']:
                            features = [[vol_feature]]
                        elif feature_names == ['volatility', 'vix']:
                            # Use current VIX (or VIX forecast if available)
                            # For simplicity, assume VIX remains constant
                            # You would need VIX forecasts for a complete implementation
                            current_vix = 20  # Placeholder
                            features = [[vol_feature, current_vix]]
                        else:
                            # Only volatility available
                            features = [[vol_feature]]

                        # Make prediction
                        forecast_spread = reg_model.predict(features)[0]
                        forecast_spreads.append(forecast_spread)

                    # Store forecasts
                    spread_forecasts[ticker] = forecast_spreads

                    # Calculate expected change
                    expected_change = forecast_spreads[-1] - current_spread if not np.isnan(current_spread) else 0
                    print(f"  {ticker}: Current={current_spread:.6f}, "
                          f"Forecast={forecast_spreads[-1]:.6f}, Δ={expected_change:.6f}")

                except Exception as e:
                    print(f"Error forecasting spread for {ticker}: {e}")
                    continue

        return spread_forecasts