import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from arch import arch_model
from scipy import stats

import config


class StatisticalModels:
    """Statistical models for analyzing and forecasting synthetic spreads"""

    def __init__(self, lookback_days: int = 252):
        """Initialize statistical models"""
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
        results = {}
        tickers = config.DATA_CONFIG['tickers']

        if not tickers:
            print("No common tickers between spreads and volatility data")
            return {}

        for ticker in tickers:
            try:
                # Prepare data
                ticker_data = pd.DataFrame({
                    'spread': spreads[ticker].dropna(),
                    'volatility': volatility[ticker].dropna()
                })

                # Add VIX if provided
                if vix is not None:
                    ticker_data['vix'] = vix.reindex(ticker_data.index).ffill()

                # Add liquidity if provided
                if liquidity is not None and ticker in liquidity.columns:
                    ticker_data['liquidity'] = liquidity[ticker].reindex(ticker_data.index).ffill()

                # Drop NaN values
                ticker_data = ticker_data.dropna()

                if len(ticker_data) < 20:
                    continue

                features = ticker_data.drop('spread', axis=1)
                target = ticker_data['spread']

                # Fit regression
                model = LinearRegression()
                model.fit(features, target)

                predictions = model.predict(features)
                residuals = target - predictions

                # Calculate R-squared
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((target - target.mean()) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                # Adjusted R-squared
                n = len(target)
                p = features.shape[1]
                adj_r_squared = 1 - (1 - r_squared) * ((n - 1) / (n - p - 1)) if n > p + 1 else r_squared

                # Store results
                results[ticker] = {
                    'model': model,
                    'r_squared': r_squared,
                    'adj_r_squared': adj_r_squared,
                    'n_observations': n,
                    'coefficients': dict(zip(features.columns, model.coef_)),
                    'intercept': model.intercept_,
                    'feature_names': features.columns.tolist(),
                    'residuals': residuals,
                    'current_prediction': model.predict([features.iloc[-1]])[0] if len(features) > 0 else np.nan,
                    'current_actual': target.iloc[-1] if len(target) > 0 else np.nan
                }

            except Exception as e:
                print(f"Error with {ticker}: {str(e)[:50]}")
                continue

        print(f"Regression complete for {len(results)} tickers")
        return results

    def calculate_regression_predictions(self,
                                         regression_results: Dict[str, Any],
                                         volatility: pd.DataFrame,
                                         vix: Optional[pd.Series] = None,
                                         liquidity: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate spread predictions using trained regression models"""
        if volatility.empty:
            return pd.DataFrame()

        predictions = pd.DataFrame(index=volatility.index)

        for ticker, results in regression_results.items():
            if ticker not in volatility.columns:
                continue

            try:
                model = results['model']
                feature_names = results['feature_names']

                # Build feature matrix
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
                    ticker_predictions = model.predict(feature_matrix)
                    predictions[ticker] = pd.Series(ticker_predictions, index=valid_dates)

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
        if tickers is None:
            tickers = returns.columns.tolist()

        results = {}

        for ticker in tickers:
            if ticker not in returns.columns:
                continue

            try:
                ticker_returns = returns[ticker].dropna()

                if len(ticker_returns) < 100:
                    continue

                # Convert to percentage
                returns_pct = ticker_returns * 100

                # Fit GARCH model
                model = arch_model(returns_pct, vol='GARCH', p=p, q=q, dist='normal')
                fitted = model.fit(disp='off', show_warning=False)

                # Get parameters
                params = fitted.params
                omega = params.get('omega', 0)
                alpha = params.get('alpha[1]', 0) if p >= 1 else 0
                beta = params.get('beta[1]', 0) if q >= 1 else 0
                persistence = alpha + beta

                # Long-run volatility
                if persistence < 1:
                    long_run_variance = omega / (1 - persistence)
                    long_run_vol = np.sqrt(long_run_variance)
                else:
                    long_run_variance = np.nan
                    long_run_vol = np.nan

                # Forecast volatility
                forecast = fitted.forecast(horizon=forecast_horizon)
                forecast_variance = forecast.variance.iloc[-1]
                forecast_vol = np.sqrt(forecast_variance) / 100

                # Current volatility
                current_vol = np.sqrt(fitted.conditional_volatility.iloc[-1]) / 100

                # Expected change
                expected_vol_change = forecast_vol.iloc[-1] - current_vol

                # Store results
                results[ticker] = {
                    'fitted_model': fitted,
                    'params': {
                        'omega': omega,
                        'alpha': alpha,
                        'beta': beta,
                        'persistence': persistence,
                        'long_run_vol': long_run_vol
                    },
                    'current_volatility': current_vol,
                    'forecast_horizon': forecast_horizon,
                    'forecast_volatility': forecast_vol.tolist(),
                    'expected_vol_change': expected_vol_change,
                    'n_observations': len(ticker_returns),
                    'aic': fitted.aic,
                    'bic': fitted.bic
                }

            except Exception as e:
                print(f"Error with {ticker}: {str(e)[:50]}")
                continue

        print(f"GARCH models fitted for {len(results)} tickers")
        return results

    def forecast_volatility_using_garch(self,
                                        returns: pd.DataFrame,
                                        garch_results: Dict[str, Any],
                                        forecast_horizon: int = 10) -> pd.DataFrame:
        """Generate volatility forecasts using pre-trained GARCH models"""
        if not garch_results:
            return pd.DataFrame()

        last_date = returns.index[-1]
        future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(forecast_horizon)]
        forecasts = pd.DataFrame(index=future_dates)

        for ticker, results in garch_results.items():
            try:
                fitted_model = results['fitted_model']
                forecast = fitted_model.forecast(horizon=forecast_horizon)
                forecast_vol = np.sqrt(forecast.variance.iloc[-1]) / 100
                forecasts[ticker] = forecast_vol.values
            except Exception as e:
                self._log(f"Error forecasting for {ticker}: {e}")
                continue

        return forecasts

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

        if isinstance(series, pd.Series):
            rolling_mean = series.rolling(window=lookback_days, min_periods=min_periods).mean()
            rolling_std = series.rolling(window=lookback_days, min_periods=min_periods).std()
            return (series - rolling_mean) / rolling_std

        elif isinstance(series, pd.DataFrame):
            zscores = pd.DataFrame(index=series.index)
            for column in series.columns:
                col_series = series[column]
                rolling_mean = col_series.rolling(window=lookback_days, min_periods=min_periods).mean()
                rolling_std = col_series.rolling(window=lookback_days, min_periods=min_periods).std()
                zscores[column] = (col_series - rolling_mean) / rolling_std
            return zscores
        else:
            raise TypeError(f"Unsupported type: {type(series)}")

    def generate_zscore_signals(self,
                                zscores: Union[pd.Series, pd.DataFrame],
                                entry_threshold: float = 2.0,
                                exit_threshold: float = 0.5) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Generate trading signals from z-scores
        
        Returns DataFrame with columns: zscore, signal, position, interpretation
        """
        def _generate_signals_for_series(zscore_series: pd.Series) -> pd.DataFrame:
            signals = pd.DataFrame(index=zscore_series.index)
            signals['zscore'] = zscore_series
            
            # Generate raw signals
            signals['signal'] = 'NEUTRAL'
            signals.loc[zscore_series < -entry_threshold, 'signal'] = 'STRONG_BUY'
            signals.loc[(zscore_series < -1.0) & (zscore_series >= -entry_threshold), 'signal'] = 'WEAK_BUY'
            signals.loc[zscore_series > entry_threshold, 'signal'] = 'STRONG_SELL'
            signals.loc[(zscore_series > 1.0) & (zscore_series <= entry_threshold), 'signal'] = 'WEAK_SELL'
            signals.loc[zscore_series.abs() < exit_threshold, 'signal'] = 'EXIT'
            
            # Generate position (numeric for backtesting)
            signals['position'] = 0
            signals.loc[zscore_series < -entry_threshold, 'position'] = 1  # Long basis
            signals.loc[zscore_series > entry_threshold, 'position'] = -1  # Short basis
            
            # Forward fill positions until exit
            signals['position'] = signals['position'].replace(0, np.nan).ffill().fillna(0)
            
            # Add interpretation
            signals['interpretation'] = signals['zscore'].apply(
                lambda z: self._interpret_zscore_value(z, entry_threshold)
            )
            
            return signals
        
        if isinstance(zscores, pd.Series):
            return _generate_signals_for_series(zscores)
        elif isinstance(zscores, pd.DataFrame):
            signals_dict = {}
            for ticker in zscores.columns:
                signals_dict[ticker] = _generate_signals_for_series(zscores[ticker])
            return signals_dict
        else:
            raise TypeError(f"Unsupported type: {type(zscores)}")

    def _interpret_zscore_value(self, zscore: float, threshold: float) -> str:
        """Get interpretation of z-score value"""
        if pd.isna(zscore):
            return "INSUFFICIENT_DATA"
        
        abs_z = abs(zscore)
        
        if zscore < -threshold:
            return "SYNTHETIC_CHEAP"
        elif zscore > threshold:
            return "SYNTHETIC_EXPENSIVE"
        elif abs_z < 0.5:
            return "FAIR_VALUE"
        elif zscore < 0:
            return "SLIGHTLY_CHEAP"
        else:
            return "SLIGHTLY_EXPENSIVE"

    def fit_arima_model(self,
                        series: pd.Series,
                        p: int = 1,
                        d: int = 1,
                        q: int = 1,
                        forecast_steps: int = 5) -> Dict[str, Any]:
        """
        Fit ARIMA model for time series forecasting
        ARIMA(p, d, q): p=autoregressive, d=differencing, q=moving average
        """
        if len(series) < 50:
            self._log(f"Series too short for ARIMA ({len(series)} points)")
            return {}

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

            # Calculate confidence intervals
            resid_std = fitted.resid.std()
            conf_int = pd.DataFrame({
                'lower': forecast - 1.96 * resid_std,
                'upper': forecast + 1.96 * resid_std
            }, index=forecast_index)

            # Determine trend
            trend = self._calculate_forecast_trend(forecast)

            return {
                'fitted_model': fitted,
                'order': (p, d, q),
                'params': fitted.params.to_dict(),
                'aic': fitted.aic,
                'bic': fitted.bic,
                'residuals': fitted.resid,
                'forecast': forecast,
                'forecast_index': forecast_index,
                'confidence_interval': conf_int,
                'last_value': series.iloc[-1],
                'forecast_trend': trend
            }

        except Exception as e:
            print(f"Error fitting ARIMA: {e}")
            return {}

    def _calculate_forecast_trend(self, forecast: pd.Series) -> str:
        """Determine trend direction from forecast"""
        if len(forecast) < 2:
            return "UNKNOWN"

        change_pct = ((forecast.iloc[-1] - forecast.iloc[0]) / forecast.iloc[0]) * 100 if forecast.iloc[0] != 0 else 0

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
        """Forecast spread trends for multiple tickers using ARIMA"""
        results = {}

        for ticker in spreads.columns:
            series = spreads[ticker].dropna()

            if len(series) < 50:
                continue

            arima_result = self.fit_arima_model(series, p, d, q, forecast_steps)

            if arima_result:
                results[ticker] = arima_result

        print(f"ARIMA forecasts complete for {len(results)} tickers")
        return results

    def hypothesis_test_cost_savings(self,
                                     savings_data: Union[pd.Series, List[float], np.ndarray],
                                     alpha: float = 0.05) -> Dict[str, Any]:
        """
        Test if savings are statistically significant
        H₀: Mean savings ≤ 0, H₁: Mean savings > 0
        """
        # Convert to array
        if isinstance(savings_data, pd.Series):
            data = savings_data.dropna().values
        elif isinstance(savings_data, list):
            data = np.array(savings_data)
        else:
            data = savings_data

        data = data[~np.isnan(data)]

        if len(data) < 2:
            return {'error': 'Need at least 2 observations'}

        # Calculate statistics
        n = len(data)
        mean_savings = np.mean(data)
        std_savings = np.std(data, ddof=1)

        # T-test
        t_stat, p_value = stats.ttest_1samp(data, 0, alternative='greater')

        # Confidence interval
        ci_lower, ci_upper = stats.t.interval(1 - alpha, n - 1,
                                              loc=mean_savings,
                                              scale=std_savings / np.sqrt(n))

        return {
            'n': n,
            'mean': mean_savings,
            'std': std_savings,
            't_stat': t_stat,
            'p_value': p_value,
            'reject_null': p_value < alpha,
            'ci_95': (ci_lower, ci_upper),
            'alpha': alpha,
            'significant': p_value < alpha
        }

    def run_full_statistical_analysis(self,
                                      market_data: Dict[str, Any],
                                      pricing_results: Dict[str, Any],
                                      lookback_days: int = 252,
                                      garch_forecast_horizon: int = 10,
                                      arima_forecast_steps: int = 5,
                                      zscore_entry_threshold: float = 2.0,
                                      zscore_exit_threshold: float = 0.5,
                                      run_regression: bool = True,
                                      run_garch: bool = True,
                                      run_zscore: bool = True,
                                      run_arima: bool = True,
                                      run_hypothesis_test: bool = True) -> Dict[str, Any]:
        """
        Run complete statistical analysis on market and pricing data
        
        Takes outputs from Section 1 (DataPipeline) and Section 2 (SyntheticPricer)
        
        Parameters:
        -----------
        market_data : Dict from DataPipeline.run_full_pipeline()
        pricing_results : Dict from SyntheticPricer.run_full_pricing_analysis()
        lookback_days : Rolling window for z-score calculation (default: 252)
        garch_forecast_horizon : Days to forecast volatility (default: 10)
        arima_forecast_steps : Steps for ARIMA forecasting (default: 5)
        zscore_entry_threshold : Z-score threshold for trade entry (default: 2.0)
        zscore_exit_threshold : Z-score threshold for trade exit (default: 0.5)
        run_regression : Whether to run regression analysis (default: True)
        run_garch : Whether to run GARCH models (default: True)
        run_zscore : Whether to calculate z-scores (default: True)
        run_arima : Whether to run ARIMA forecasting (default: True)
        run_hypothesis_test : Whether to run hypothesis test (default: True)
        
        Returns:
        --------
        Dictionary containing:
            - regression_results: Dict with regression models per ticker
            - spread_predictions: DataFrame with predicted spreads
            - garch_results: Dict with GARCH models per ticker
            - volatility_forecasts: DataFrame with volatility forecasts
            - zscores: DataFrame with z-scores for basis
            - zscore_signals: Dict with trading signals per ticker
            - arima_results: Dict with ARIMA forecasts per ticker
            - hypothesis_test: Dict with statistical test results
            - summary: Dict with aggregate statistics
        """
        
        print("\nRunning full statistical analysis...")
        print(f"Lookback: {lookback_days} days, GARCH horizon: {garch_forecast_horizon} days")
        
        # Extract data
        prices = market_data['prices']
        returns = market_data['returns']
        volatilities = market_data['volatility']
        vix = market_data.get('vix')
        liquidity = market_data.get('liquidity')
        
        historical_basis = pricing_results.get('historical_basis', pd.DataFrame())
        current_analysis = pricing_results.get('current_analysis', pd.DataFrame())
        spread_stats = pricing_results.get('spread_stats', pd.DataFrame())
        
        tickers = list(volatilities.columns)
        
        # Prepare spreads DataFrame from spread_stats
        spreads = spread_stats[['current_spread']].T if not spread_stats.empty else pd.DataFrame()
        if not spreads.empty:
            spreads.index = volatilities.index[-1:]
            spreads = spreads.reindex(volatilities.index, method='ffill')
        
        results = {}
        
        # Regression Analysis
        regression_results = {}
        spread_predictions = pd.DataFrame()
        if run_regression and not spreads.empty:
            print("Running regression analysis...")
            regression_results = self.regression_spread_drivers(
                spreads=spreads,
                volatility=volatilities,
                vix=vix,
                liquidity=liquidity
            )
            
            if regression_results:
                spread_predictions = self.calculate_regression_predictions(
                    regression_results=regression_results,
                    volatility=volatilities,
                    vix=vix,
                    liquidity=liquidity
                )
                print(f"Regression models fitted for {len(regression_results)} tickers")
        
        # GARCH Volatility Forecasting
        garch_results = {}
        volatility_forecasts = pd.DataFrame()
        if run_garch:
            print("Fitting GARCH models...")
            garch_results = self.fit_garch_volatility(
                returns=returns,
                tickers=tickers,
                p=1,
                q=1,
                forecast_horizon=garch_forecast_horizon
            )
            
            if garch_results:
                volatility_forecasts = self.forecast_volatility_using_garch(
                    returns=returns,
                    garch_results=garch_results,
                    forecast_horizon=garch_forecast_horizon
                )
                print(f"GARCH models fitted for {len(garch_results)} tickers")
        
        # Z-Score Analysis and Signal Generation
        zscores = pd.DataFrame()
        zscore_signals = {}
        if run_zscore and not historical_basis.empty:
            print("Calculating z-scores and generating signals...")
            
            # Pivot basis data
            basis_pivot = historical_basis.pivot(index='date', columns='ticker', values='basis')
            
            # Calculate z-scores
            zscores = self.calculate_zscore(
                series=basis_pivot,
                lookback_days=lookback_days
            )
            
            # Generate trading signals
            zscore_signals = self.generate_zscore_signals(
                zscores=zscores,
                entry_threshold=zscore_entry_threshold,
                exit_threshold=zscore_exit_threshold
            )
            
            print(f"Z-scores calculated for {len(zscores.columns)} tickers")
        
        # ARIMA Forecasting
        arima_results = {}
        if run_arima and not spreads.empty:
            print("Running ARIMA forecasting...")
            arima_results = self.forecast_spread_trends(
                spreads=spreads,
                p=1,
                d=1,
                q=1,
                forecast_steps=arima_forecast_steps
            )
            print(f"ARIMA forecasts generated for {len(arima_results)} tickers")
        
        # Hypothesis Test
        hypothesis_test = {}
        if run_hypothesis_test and not current_analysis.empty and 'savings_vs_long' in current_analysis.columns:
            print("Running hypothesis test...")
            hypothesis_test = self.hypothesis_test_cost_savings(
                savings_data=current_analysis['savings_vs_long']
            )
            
            if hypothesis_test.get('significant'):
                print(f"Savings are statistically significant (p={hypothesis_test['p_value']:.4f})")
            else:
                print(f"Savings not statistically significant (p={hypothesis_test['p_value']:.4f})")
        
        # Create summary
        summary = {
            'num_tickers': len(tickers),
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'lookback_days': lookback_days,
            'models_fitted': {
                'regression': len(regression_results),
                'garch': len(garch_results),
                'arima': len(arima_results),
                'zscore_signals': len(zscore_signals)
            }
        }
        
        # Add regression summary
        if regression_results:
            avg_r2 = np.mean([r['r_squared'] for r in regression_results.values()])
            summary['avg_r_squared'] = avg_r2
        
        # Add GARCH summary
        if garch_results:
            avg_persistence = np.mean([r['params']['persistence'] for r in garch_results.values()])
            summary['avg_garch_persistence'] = avg_persistence
        
        # Add z-score summary
        if not zscores.empty:
            current_zscores = zscores.iloc[-1].dropna()
            summary['current_zscores'] = {
                'mean': current_zscores.mean(),
                'extreme_count': (current_zscores.abs() > zscore_entry_threshold).sum(),
                'max_abs': current_zscores.abs().max()
            }
        
        # Add hypothesis test summary
        if hypothesis_test and not hypothesis_test.get('error'):
            summary['hypothesis_test'] = {
                'mean_savings': hypothesis_test['mean'],
                'significant': hypothesis_test['significant'],
                'p_value': hypothesis_test['p_value']
            }
        
        # Package results
        results = {
            'regression_results': regression_results,
            'spread_predictions': spread_predictions,
            'garch_results': garch_results,
            'volatility_forecasts': volatility_forecasts,
            'zscores': zscores,
            'zscore_signals': zscore_signals,
            'arima_results': arima_results,
            'hypothesis_test': hypothesis_test,
            'summary': summary
        }
        
        # Print summary
        print("\nSTATISTICAL ANALYSIS COMPLETE")
        print(f"Tickers analyzed: {summary['num_tickers']}")
        print(f"Models fitted: {summary['models_fitted']}")
        
        if 'avg_r_squared' in summary:
            print(f"Average R²: {summary['avg_r_squared']:.3f}")
        
        if 'avg_garch_persistence' in summary:
            print(f"Average GARCH persistence: {summary['avg_garch_persistence']:.3f}")
        
        if 'current_zscores' in summary:
            print(f"Current z-scores: mean={summary['current_zscores']['mean']:.2f}, "
                  f"extreme={summary['current_zscores']['extreme_count']}")
        
        if 'hypothesis_test' in summary:
            print(f"Mean savings: {summary['hypothesis_test']['mean_savings']:.6f} "
                  f"({'significant' if summary['hypothesis_test']['significant'] else 'not significant'})")
        
        print()
        
        return results