import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from arch import  arch_model
from scipy import stats

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

    def calculate_zscore(self,
                         series: Union[pd.Series, pd.DataFrame],
                         lookback_days: int = None,
                         min_periods: int = 20) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate Z-scores to identify abnormal spread levels

        Formula: Z = (Current Value - Rolling Mean) / Rolling Std Dev
        """
        if lookback_days is None:
            lookback_days = self.lookback_days

        print(f"Calculating Z-scores ({lookback_days}-day lookback)")

        if isinstance(series, pd.Series):
            rolling_mean = series.rolling(window=lookback_days, min_periods=min_periods).mean()
            rolling_std = series.rolling(window=lookback_days, min_periods=min_periods).std()
            zscore = (series - rolling_mean) / rolling_std

            print(f"  {series.name or 'Series'}: mean={rolling_mean.iloc[-1]:.6f}, "
                  f"std={rolling_std.iloc[-1]:.6f}, current Z={zscore.iloc[-1]:.2f}")

            return zscore

        elif isinstance(series, pd.DataFrame):
            zscores = pd.DataFrame(index=series.index)

            for column in series.columns:
                col_series = series[column]
                rolling_mean = col_series.rolling(window=lookback_days, min_periods=min_periods).mean()
                rolling_std = col_series.rolling(window=lookback_days, min_periods=min_periods).std()
                zscores[column] = (col_series - rolling_mean) / rolling_std

                if not col_series.empty:
                    print(f"  {column}: mean={rolling_mean.iloc[-1]:.6f}, "
                          f"std={rolling_std.iloc[-1]:.6f}, current Z={zscores[column].iloc[-1]:.2f}")

            return zscores

        else:
            raise TypeError(f"Unsupported type: {type(series)}")

    def interpret_zscore(self,
                         zscore: float,
                         threshold: float = 2.0) -> Dict[str, Any]:
        """
        Interpret Z-score for trading decisions
        """
        abs_z = abs(zscore)

        if abs_z > threshold:
            significance = "SIGNIFICANT"
            if zscore < -threshold:
                direction = "BELOW"
                signal = "LONG_BASIS"  # Synthetic is cheap
            else:
                direction = "ABOVE"
                signal = "SHORT_BASIS"  # Synthetic is expensive
        else:
            significance = "NORMAL"
            direction = "WITHIN"
            signal = "NEUTRAL"

        # Calculate probability (assuming normal distribution)
        from scipy import stats
        prob_extreme = 2 * (1 - stats.norm.cdf(abs_z))  # Two-tailed probability

        interpretation = {
            'zscore': zscore,
            'abs_zscore': abs_z,
            'significance': significance,
            'direction': direction,
            'signal': signal,
            'threshold': threshold,
            'prob_extreme': prob_extreme,
            'interpretation': self._get_zscore_interpretation(zscore, threshold)
        }

        return interpretation

    def _get_zscore_interpretation(self, zscore: float, threshold: float) -> str:
        """Get human-readable interpretation of Z-score"""
        abs_z = abs(zscore)

        if abs_z > 2.5:
            level = "EXTREMELY"
        elif abs_z > 2.0:
            level = "VERY"
        elif abs_z > 1.5:
            level = "MODERATELY"
        elif abs_z > 1.0:
            level = "SLIGHTLY"
        else:
            level = "WITHIN"

        if zscore < -threshold:
            return f"{level} BELOW average (synthetic appears CHEAP)"
        elif zscore > threshold:
            return f"{level} ABOVE average (synthetic appears EXPENSIVE)"
        else:
            return f"{level} normal range"

    def fit_arima_model(self,
                        series: pd.Series,
                        p: int = 1,
                        d: int = 1,
                        q: int = 1,
                        forecast_steps: int = 5) -> Dict[str, Any]:
        """
        Fit ARIMA model for time series forecasting

        ARIMA(p, d, q):
        - p = autoregressive terms
        - d = differencing order
        - q = moving average terms
        """

        if len(series) < 50:
            print(f"Warning: Series too short for ARIMA ({len(series)} points)")
            return {}

        print(f"Fitting ARIMA({p},{d},{q}) model...")
        print(f"  Series: {series.name or 'unnamed'}, {len(series)} points")

        try:
            # Fit ARIMA model
            model = ARIMA(series, order=(p, d, q))
            fitted = model.fit()

            # Generate forecast
            forecast = fitted.forecast(steps=forecast_steps)
            forecast_index = pd.date_range(
                start=series.index[-1] + pd.Timedelta(days=1),
                periods=forecast_steps,
                freq='B'
            )

            # Get model summary
            params = fitted.params
            aic = fitted.aic
            bic = fitted.bic
            resid = fitted.resid

            # Calculate forecast confidence intervals (simplified)
            resid_std = resid.std()
            conf_int = pd.DataFrame({
                'lower': forecast - 1.96 * resid_std,
                'upper': forecast + 1.96 * resid_std
            }, index=forecast_index)

            # Store results
            results = {
                'fitted_model': fitted,
                'order': (p, d, q),
                'params': params.to_dict(),
                'aic': aic,
                'bic': bic,
                'residuals': resid,
                'residual_std': resid_std,
                'forecast': forecast,
                'forecast_index': forecast_index,
                'confidence_interval': conf_int,
                'last_value': series.iloc[-1],
                'forecast_steps': forecast_steps,
                'forecast_trend': self._calculate_forecast_trend(forecast)
            }

            # Print summary
            print(f"  Model fit: AIC={aic:.2f}, BIC={bic:.2f}")
            print(f"  Last value: {series.iloc[-1]:.6f}")
            print(f"  {forecast_steps}-step forecast: {forecast.iloc[-1]:.6f}")
            print(f"  Forecast trend: {results['forecast_trend']}")

            return results

        except Exception as e:
            print(f"Error fitting ARIMA model: {e}")
            return {}

    def _calculate_forecast_trend(self, forecast: pd.Series) -> str:
        """Determine trend direction from forecast"""
        if len(forecast) < 2:
            return "UNKNOWN"

        start = forecast.iloc[0]
        end = forecast.iloc[-1]
        change = end - start
        change_pct = (change / start) * 100 if start != 0 else 0

        if change_pct > 1.0:
            return "STRONGLY_UP"
        elif change_pct > 0.2:
            return "MODERATELY_UP"
        elif change_pct > -0.2:
            return "FLAT"
        elif change_pct > -1.0:
            return "MODERATELY_DOWN"
        else:
            return "STRONGLY_DOWN"

    def forecast_spread_trends(self,
                               spreads: pd.DataFrame,
                               p: int = 1,
                               d: int = 1,
                               q: int = 1,
                               forecast_steps: int = 5) -> Dict[str, Any]:
        """
        Forecast spread trends for multiple tickers using ARIMA
        """
        results = {}

        print(f"Forecasting spread trends for {len(spreads.columns)} tickers")
        print(f"ARIMA({p},{d},{q}), {forecast_steps}-step forecast")

        for ticker in spreads.columns:
            series = spreads[ticker].dropna()

            if len(series) < 50:
                print(f"  {ticker}: Skipped (insufficient data: {len(series)} points)")
                continue

            print(f"  {ticker}...", end="")
            arima_result = self.fit_arima_model(series, p, d, q, forecast_steps)

            if arima_result:
                results[ticker] = arima_result
                current = series.iloc[-1]
                forecast = arima_result['forecast'].iloc[-1]
                change = forecast - current
                change_bps = change * 10000

                print(f" done: {current:.6f} → {forecast:.6f} (Δ={change_bps:+.1f} bps)")
            else:
                print(f" failed")

        # Print summary
        if results:
            print(f"\nForecast Summary:")
            print(f"{'Ticker':<8} {'Current':<12} {'Forecast':<12} {'Change':<12} {'Trend':<15}")
            print("-" * 60)

            for ticker, res in results.items():
                current = res['last_value']
                forecast = res['forecast'].iloc[-1]
                change = forecast - current
                change_bps = change * 10000
                trend = res['forecast_trend']

                print(f"{ticker:<8} {current:.6f}    {forecast:.6f}    {change_bps:+.1f} bps     {trend}")

        return results

    def compare_arima_orders(self,
                             series: pd.Series,
                             orders: List[Tuple[int, int, int]] = None,
                             forecast_steps: int = 5) -> pd.DataFrame:
        """
        Compare different ARIMA orders to find best fit
        """
        if orders is None:
            orders = [
                (1, 1, 1),  # Basic ARIMA
                (2, 1, 2),  # More complex
                (0, 1, 1),  # Simple MA
                (1, 0, 0),  # Simple AR
            ]

        print(f"Comparing {len(orders)} ARIMA orders...")

        comparison = []

        for p, d, q in orders:
            try:
                result = self.fit_arima_model(series, p, d, q, forecast_steps)

                if result:
                    comparison.append({
                        'order': f"({p},{d},{q})",
                        'p': p,
                        'd': d,
                        'q': q,
                        'aic': result['aic'],
                        'bic': result['bic'],
                        'forecast': result['forecast'].iloc[-1],
                        'trend': result['forecast_trend'],
                        'residual_std': result['residual_std']
                    })

            except Exception as e:
                print(f"  ARIMA({p},{d},{q}) failed: {e}")
                continue

        if not comparison:
            return pd.DataFrame()

        comparison_df = pd.DataFrame(comparison)

        # Sort by AIC (lower is better)
        comparison_df = comparison_df.sort_values('aic')

        print(f"\nARIMA Order Comparison:")
        print(f"{'Order':<10} {'AIC':<12} {'BIC':<12} {'Forecast':<12} {'Trend':<15}")
        print("-" * 60)

        for _, row in comparison_df.iterrows():
            print(f"{row['order']:<10} {row['aic']:.2f}    {row['bic']:.2f}    "
                  f"{row['forecast']:.6f}    {row['trend']}")

        best_order = comparison_df.iloc[0]
        print(f"\nBest model: ARIMA{best_order['order']} (AIC={best_order['aic']:.2f})")

        return comparison_df

    def hypothesis_test_cost_savings(self,
                                     savings_data: Union[pd.Series, List[float], np.ndarray],
                                     alpha: float = 0.05) -> Dict[str, Any]:
        """
        Simple hypothesis test: Are savings statistically significant?

        H₀: Mean savings ≤ 0 (no benefit)
        H₁: Mean savings > 0 (real benefit)
        """
        # Convert to array and clean
        if isinstance(savings_data, pd.Series):
            data = savings_data.dropna().values
        elif isinstance(savings_data, list):
            data = np.array(savings_data)
        else:
            data = savings_data

        data = data[~np.isnan(data)]

        if len(data) < 2:
            return {'error': 'Need at least 2 observations'}

        # Basic stats
        n = len(data)
        mean_savings = np.mean(data)
        std_savings = np.std(data, ddof=1)

        # t-test
        t_stat, p_value = stats.ttest_1samp(data, 0, alternative='greater')

        # Confidence interval
        ci_lower, ci_upper = stats.t.interval(1 - alpha, n - 1,
                                              loc=mean_savings,
                                              scale=std_savings / np.sqrt(n))

        # Result
        reject_h0 = p_value < alpha

        return {
            'n': n,
            'mean': mean_savings,
            'std': std_savings,
            't_stat': t_stat,
            'p_value': p_value,
            'reject_null': reject_h0,
            'ci_95': (ci_lower, ci_upper),
            'alpha': alpha
        }

    def print_hypothesis_test_summary(self, results: Dict[str, Any]):
        """Simple print of hypothesis test results"""
        if 'error' in results:
            print(f"Error: {results['error']}")
            return

        print(f"\nHypothesis Test Results:")
        print(f"  Observations: {results['n']}")
        print(f"  Mean savings: {results['mean']:.6f} ({results['mean'] * 10000:.1f} bps)")
        print(f"  t-statistic: {results['t_stat']:.2f}")
        print(f"  p-value: {results['p_value']:.4f}")

        ci_lower, ci_upper = results['ci_95']
        print(f"  95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")

        print(f"\nConclusion:")
        if results['reject_null']:
            print(f"  REJECT null hypothesis (p < {results['alpha']})")
            print(f"  Savings are statistically significant")
        else:
            print(f"  Cannot reject null hypothesis (p ≥ {results['alpha']})")
            print(f"  No evidence of significant savings")