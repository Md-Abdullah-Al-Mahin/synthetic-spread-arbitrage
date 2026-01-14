import pandas as pd
import numpy as np
from typing import Dict, Union, Optional, Any, List
from scipy import stats
import config


class RiskAnalytics:
    """Risk analytics for synthetic spread arbitrage strategy"""
    
    def __init__(self):
        """Initialize risk analytics"""
        pass
    
    def calculate_var(self,
                     returns: Union[pd.Series, pd.DataFrame, np.ndarray],
                     confidence_level: float = 0.95,
                     method: str = 'historical') -> Union[float, pd.Series]:
        """
        Calculate Value at Risk (VaR)
        VaR answers: "What's the worst loss I can expect X% of the time?"
        """
        return self._apply_to_returns(returns, self._var_array, confidence_level, method)
    
    def _var_array(self, returns_array: np.ndarray, confidence_level: float, method: str) -> float:
        """Core VaR calculation on array"""
        if len(returns_array) == 0:
            return np.nan
        
        if method == 'historical':
            return np.percentile(returns_array, (1 - confidence_level) * 100)
        else:  # parametric
            mean = np.mean(returns_array)
            std = np.std(returns_array)
            z_score = stats.norm.ppf(1 - confidence_level)
            return mean + z_score * std
    
    def calculate_cvar(self,
                      returns: Union[pd.Series, pd.DataFrame, np.ndarray],
                      confidence_level: float = 0.95,
                      method: str = 'historical') -> Union[float, pd.Series]:
        """
        Calculate Conditional Value at Risk (CVaR)
        CVaR answers: "When losses exceed VaR, what's the average loss?"
        """
        return self._apply_to_returns(returns, self._cvar_array, confidence_level, method)
    
    def _cvar_array(self, returns_array: np.ndarray, confidence_level: float, method: str) -> float:
        """Core CVaR calculation on array"""
        if len(returns_array) == 0:
            return np.nan
        
        if method == 'historical':
            var_threshold = self._var_array(returns_array, confidence_level, method)
            tail_returns = returns_array[returns_array <= var_threshold]
            return tail_returns.mean() if len(tail_returns) > 0 else var_threshold
        else:  # parametric
            mean = np.mean(returns_array)
            std = np.std(returns_array)
            z_alpha = stats.norm.ppf(1 - confidence_level)
            phi_z = stats.norm.pdf(z_alpha)
            return mean - std * phi_z / (1 - confidence_level)
    
    def compare_var_cvar(self,
                        returns: Union[pd.Series, pd.DataFrame],
                        position_value: float = None,
                        confidence_level: float = 0.95,
                        method: str = 'historical') -> Dict[str, Any]:
        """Compare VaR and CVaR to assess tail risk"""
        var_pct = self.calculate_var(returns, confidence_level, method)
        cvar_pct = self.calculate_cvar(returns, confidence_level, method)
        
        if isinstance(var_pct, pd.Series):
            tail_risk_pct = cvar_pct - var_pct
            tail_risk_ratio = cvar_pct / var_pct
            
            result = {
                'var_pct': var_pct,
                'cvar_pct': cvar_pct,
                'tail_risk_pct': tail_risk_pct,
                'tail_risk_ratio': tail_risk_ratio,
                'confidence_level': confidence_level,
                'method': method
            }
            
            if position_value:
                result['var_dollar'] = var_pct * position_value
                result['cvar_dollar'] = cvar_pct * position_value
                result['tail_risk_dollar'] = tail_risk_pct * position_value
        else:
            tail_risk_pct = cvar_pct - var_pct
            tail_risk_ratio = cvar_pct / var_pct if var_pct != 0 else np.nan
            
            result = {
                'var_pct': var_pct,
                'cvar_pct': cvar_pct,
                'tail_risk_pct': tail_risk_pct,
                'tail_risk_ratio': tail_risk_ratio,
                'confidence_level': confidence_level,
                'method': method
            }
            
            if position_value:
                result['var_dollar'] = var_pct * position_value
                result['cvar_dollar'] = cvar_pct * position_value
                result['tail_risk_dollar'] = tail_risk_pct * position_value
        
        return result
    
    def monte_carlo_simulation(self,
                              returns: Union[pd.Series, np.ndarray],
                              initial_value: float = 1000000,
                              n_simulations: int = 10000,
                              time_horizon: int = 252,
                              mean_adjustment: float = 0.0,
                              vol_multiplier: float = 1.0,
                              daily_cost: float = 0.0,
                              random_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Core Monte Carlo simulation function
        
        All scenario and stress testing functions call this
        
        Parameters:
        -----------
        returns : Historical returns
        initial_value : Starting value
        n_simulations : Number of paths
        time_horizon : Days to simulate
        mean_adjustment : Shift mean return (for scenarios)
        vol_multiplier : Scale volatility (for scenarios)
        daily_cost : Daily financing/spread cost (for scenarios)
        random_seed : Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Convert to array
        if isinstance(returns, pd.Series):
            returns_array = returns.dropna().values
        else:
            returns_array = returns[~np.isnan(returns)]
        
        if len(returns_array) == 0:
            return {'error': 'No valid returns data'}
        
        # Calculate adjusted parameters
        base_mean = np.mean(returns_array)
        base_std = np.std(returns_array)
        
        scenario_mean = base_mean + mean_adjustment
        scenario_std = base_std * vol_multiplier
        
        # Run simulations
        sim_values = np.zeros((n_simulations, time_horizon + 1))
        sim_values[:, 0] = initial_value
        
        for i in range(n_simulations):
            for t in range(1, time_horizon + 1):
                random_return = np.random.normal(scenario_mean, scenario_std)
                sim_values[i, t] = sim_values[i, t-1] * (1 + random_return) - daily_cost
        
        # Calculate statistics
        final_values = sim_values[:, -1]
        sim_returns = (final_values - initial_value) / initial_value
        
        mean_final = np.mean(final_values)
        var_95 = np.percentile(sim_returns, 5)
        cvar_95 = np.mean(sim_returns[sim_returns <= var_95])
        
        return {
            'initial_value': initial_value,
            'n_simulations': n_simulations,
            'time_horizon': time_horizon,
            'mean_final': mean_final,
            'median_final': np.median(final_values),
            'std_final': np.std(final_values),
            'var_95': var_95 * initial_value,
            'cvar_95': cvar_95 * initial_value,
            'prob_loss': np.mean(final_values < initial_value),
            'prob_gain_10pct': np.mean(final_values >= initial_value * 1.10),
            'best_case': np.max(final_values),
            'worst_case': np.min(final_values),
            'simulation_paths': sim_values,
            'final_values': final_values
        }
    
    def scenario_analysis(self,
                         returns: Union[pd.Series, pd.DataFrame],
                         initial_value: float,
                         scenarios: Dict[str, Dict[str, float]],
                         n_simulations: int = 10000,
                         time_horizon: int = 252,
                         current_volatility: Optional[float] = None,
                         current_spread: float = 0.0020,
                         spread_vol_coefficient: float = 0.002,
                         random_seed: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        Run Monte Carlo under different scenario assumptions
        
        Each scenario specifies shocks that adjust the simulation parameters
        """
        # Get returns array
        if isinstance(returns, pd.Series):
            returns_array = returns.dropna().values
        else:
            returns_array = returns.mean(axis=1).dropna().values
        
        # Calculate current volatility
        if current_volatility is None:
            current_volatility = np.std(returns_array) * np.sqrt(252)
        
        results = {}
        
        for scenario_name, shocks in scenarios.items():
            # Extract shocks
            price_shock = shocks.get('price_shock', 0.0)
            vol_shock = shocks.get('vol_shock', 1.0)
            spread_shock_bps = shocks.get('spread_shock', 0.0)
            
            # Calculate new spread
            new_volatility = current_volatility * vol_shock
            vol_change = new_volatility - current_volatility
            spread_change_from_vol = vol_change * spread_vol_coefficient
            total_spread_change = (spread_change_from_vol * 10000) + spread_shock_bps
            new_spread = current_spread + (total_spread_change / 10000)
            
            # Daily cost from spread
            daily_cost = (new_spread / 252) * initial_value
            
            # Run Monte Carlo with adjusted parameters
            mc_results = self.monte_carlo_simulation(
                returns=returns_array,
                initial_value=initial_value,
                n_simulations=n_simulations,
                time_horizon=time_horizon,
                mean_adjustment=price_shock / time_horizon,
                vol_multiplier=vol_shock,
                daily_cost=daily_cost,
                random_seed=random_seed
            )
            
            # Add scenario-specific info
            mc_results['scenario_params'] = {
                'price_shock': price_shock,
                'vol_shock': vol_shock,
                'spread_shock_bps': spread_shock_bps,
                'new_volatility': new_volatility,
                'new_spread_bps': new_spread * 10000
            }
            
            results[scenario_name] = mc_results
        
        return results
    
    def create_standard_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Standard stress test scenarios"""
        return config.STRESS_TEST_SCENARIOS
    
    def compare_scenarios(self, scenario_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Compare scenarios side-by-side"""
        data = []
        
        for name, results in scenario_results.items():
            params = results.get('scenario_params', {})
            data.append({
                'Scenario': name,
                'Price Shock': params.get('price_shock', 0),
                'Vol Shock': params.get('vol_shock', 1),
                'Mean Final': results['mean_final'],
                'VaR(95%)': results['var_95'],
                'CVaR(95%)': results['cvar_95'],
                'Prob Loss': results['prob_loss'],
                'Worst Case': results['worst_case']
            })
        
        return pd.DataFrame(data).set_index('Scenario').sort_values('Mean Final')
    
    def _apply_to_returns(self, returns, func, *args):
        """Apply function to returns, handling different input types"""
        if isinstance(returns, pd.Series):
            clean = returns.dropna().values
            return func(clean, *args)
        
        elif isinstance(returns, pd.DataFrame):
            results = {}
            for col in returns.columns:
                clean = returns[col].dropna().values
                results[col] = func(clean, *args)
            return pd.Series(results)
        
        elif isinstance(returns, np.ndarray):
            clean = returns[~np.isnan(returns)]
            return func(clean, *args)
        
        else:
            raise TypeError(f"Unsupported type: {type(returns)}")

    def calculate_sharpe_ratio(self,
                            returns: Union[pd.Series, pd.DataFrame, np.ndarray],
                            risk_free_rate: float = 0.045,
                            periods_per_year: int = 252) -> Union[float, pd.Series]:
        """
        Calculate Sharpe Ratio: (Annualized Return - Risk Free) / Annualized Volatility
        
        Interpretation: >1.0 Good, >1.5 Very Good, >2.0 Excellent
        """
        return self._apply_to_returns(returns, self._sharpe_array, risk_free_rate, periods_per_year)

    def _sharpe_array(self, returns_array: np.ndarray, risk_free_rate: float, periods_per_year: int) -> float:
        """Core Sharpe calculation"""
        if len(returns_array) == 0:
            return np.nan
        
        annual_return = np.mean(returns_array) * periods_per_year
        annual_vol = np.std(returns_array, ddof=1) * np.sqrt(periods_per_year)
        
        return (annual_return - risk_free_rate) / annual_vol if annual_vol != 0 else np.nan

    def calculate_sortino_ratio(self,
                            returns: Union[pd.Series, pd.DataFrame, np.ndarray],
                            risk_free_rate: float = 0.045,
                            target_return: float = 0.0,
                            periods_per_year: int = 252) -> Union[float, pd.Series]:
        """
        Calculate Sortino Ratio: like Sharpe but only penalizes downside volatility
        
        Higher Sortino vs Sharpe = asymmetric returns (good!)
        """
        return self._apply_to_returns(returns, self._sortino_array, risk_free_rate, target_return, periods_per_year)

    def _sortino_array(self, returns_array: np.ndarray, risk_free_rate: float, 
                    target_return: float, periods_per_year: int) -> float:
        """Core Sortino calculation"""
        if len(returns_array) == 0:
            return np.nan
        
        annual_return = np.mean(returns_array) * periods_per_year
        
        # Downside deviation only
        downside_returns = returns_array[returns_array < target_return] - target_return
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        annual_downside_dev = downside_std * np.sqrt(periods_per_year)
        
        return (annual_return - risk_free_rate) / annual_downside_dev if annual_downside_dev != 0 else np.nan

    def compare_sharpe_sortino(self,
                            returns: Union[pd.Series, pd.DataFrame],
                            risk_free_rate: float = 0.045,
                            periods_per_year: int = 252) -> Union[Dict[str, Any], pd.DataFrame]:
        """Compare Sharpe and Sortino to assess risk profile"""
        sharpe = self.calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
        sortino = self.calculate_sortino_ratio(returns, risk_free_rate, 0.0, periods_per_year)
        
        if isinstance(sharpe, pd.Series):
            # Multiple assets
            return pd.DataFrame({
                'Sharpe': sharpe,
                'Sortino': sortino,
                'Sortino/Sharpe': sortino / sharpe,
                'Risk Profile': (sortino / sharpe).apply(self._interpret_sortino_sharpe_ratio)
            }).sort_values('Sortino', ascending=False)
        else:
            # Single asset
            ratio = sortino / sharpe if sharpe != 0 else np.nan
            return {
                'sharpe': sharpe,
                'sortino': sortino,
                'sortino_sharpe_ratio': ratio,
                'risk_profile': self._interpret_sortino_sharpe_ratio(ratio)
            }

    def _interpret_sortino_sharpe_ratio(self, ratio: float) -> str:
        """Interpret Sortino/Sharpe ratio"""
        if np.isnan(ratio) or np.isinf(ratio):
            return "UNKNOWN"
        elif ratio > 1.3:
            return "EXCELLENT (Asymmetric upside)"
        elif ratio > 1.15:
            return "GOOD (Limited downside)"
        elif ratio > 1.0:
            return "ACCEPTABLE"
        else:
            return "SYMMETRIC (High downside)"

    def calculate_max_drawdown(self,
                            returns: Union[pd.Series, pd.DataFrame, np.ndarray],
                            initial_value: float = 1.0) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Calculate Maximum Drawdown: worst peak-to-trough decline
        
        Interpretation: <5% Minimal, 5-10% Acceptable, 10-20% Moderate, >20% High
        """
        if isinstance(returns, pd.Series):
            return self._max_drawdown_series(returns, initial_value)
        
        elif isinstance(returns, pd.DataFrame):
            results = {col: self._max_drawdown_series(returns[col], initial_value) 
                    for col in returns.columns}
            return pd.DataFrame(results).T
        
        elif isinstance(returns, np.ndarray):
            temp_series = pd.Series(returns)
            result = self._max_drawdown_series(temp_series, initial_value)
            result.pop('peak_date', None)
            result.pop('trough_date', None)
            result.pop('recovery_date', None)
            return result
        
        else:
            raise TypeError(f"Unsupported type: {type(returns)}")

    def _max_drawdown_series(self, returns: pd.Series, initial_value: float) -> Dict[str, Any]:
        """Core max drawdown calculation"""
        clean_returns = returns.dropna()
        
        if len(clean_returns) == 0:
            return self._empty_drawdown_result()
        
        # Cumulative portfolio value
        cumulative = (1 + clean_returns).cumprod() * initial_value
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Find max drawdown
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        peak_idx = cumulative[:max_dd_idx].idxmax()
        
        # Find recovery
        after_trough = cumulative[max_dd_idx:]
        peak_value = cumulative[peak_idx]
        recovered = after_trough[after_trough >= peak_value]
        recovery_date = recovered.index[0] if len(recovered) > 0 else None
        
        # Duration
        duration = self._calculate_duration(peak_idx, max_dd_idx)
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_dollar': cumulative[max_dd_idx] - cumulative[peak_idx],
            'peak_value': cumulative[peak_idx],
            'trough_value': cumulative[max_dd_idx],
            'peak_date': peak_idx,
            'trough_date': max_dd_idx,
            'duration_days': duration,
            'recovery_date': recovery_date,
            'recovered': recovery_date is not None,
            'drawdown_series': drawdown
        }

    def _empty_drawdown_result(self) -> Dict[str, Any]:
        """Return empty drawdown result"""
        return {
            'max_drawdown': np.nan,
            'max_drawdown_dollar': np.nan,
            'peak_date': None,
            'trough_date': None,
            'duration_days': 0,
            'recovery_date': None,
            'recovered': False
        }

    def _calculate_duration(self, start_idx, end_idx) -> int:
        """Calculate duration between two indices"""
        if isinstance(start_idx, pd.Timestamp) and isinstance(end_idx, pd.Timestamp):
            return (end_idx - start_idx).days
        elif isinstance(end_idx, int):
            return end_idx - start_idx
        else:
            return 0

    def calculate_drawdown_statistics(self,
                                    returns: Union[pd.Series, pd.DataFrame],
                                    initial_value: float = 1.0) -> Union[Dict[str, Any], pd.DataFrame]:
        """Calculate comprehensive drawdown statistics"""
        if isinstance(returns, pd.Series):
            return self._drawdown_statistics_series(returns, initial_value)
        
        elif isinstance(returns, pd.DataFrame):
            results = {col: self._drawdown_statistics_series(returns[col], initial_value) 
                    for col in returns.columns}
            return pd.DataFrame(results).T
        
        else:
            raise TypeError(f"Unsupported type: {type(returns)}")

    def _drawdown_statistics_series(self, returns: pd.Series, initial_value: float) -> Dict[str, Any]:
        """Calculate drawdown statistics for single series"""
        clean_returns = returns.dropna()
        
        if len(clean_returns) == 0:
            return {
                'max_drawdown': np.nan,
                'avg_drawdown': np.nan,
                'current_drawdown': np.nan
            }
        
        dd_info = self._max_drawdown_series(returns, initial_value)
        drawdown_series = dd_info['drawdown_series']
        
        negative_dd = drawdown_series[drawdown_series < 0]
        
        return {
            'max_drawdown': dd_info['max_drawdown'],
            'avg_drawdown': negative_dd.mean() if len(negative_dd) > 0 else 0.0,
            'drawdown_frequency': len(negative_dd) / len(drawdown_series),
            'current_drawdown': drawdown_series.iloc[-1]
        }

    def calculate_calmar_ratio(self,
                            returns: Union[pd.Series, pd.DataFrame],
                            risk_free_rate: float = 0.045,
                            periods_per_year: int = 252) -> Union[float, pd.Series]:
        """Calculate Calmar Ratio: Return / Max Drawdown"""
        if isinstance(returns, pd.Series):
            return self._calmar_ratio_series(returns, risk_free_rate, periods_per_year)
        
        elif isinstance(returns, pd.DataFrame):
            return pd.Series({col: self._calmar_ratio_series(returns[col], risk_free_rate, periods_per_year) 
                            for col in returns.columns}, name='Calmar_Ratio')
        
        else:
            raise TypeError(f"Unsupported type: {type(returns)}")

    def _calmar_ratio_series(self, returns: pd.Series, risk_free_rate: float, 
                            periods_per_year: int) -> float:
        """Core Calmar calculation"""
        clean_returns = returns.dropna()
        
        if len(clean_returns) == 0:
            return np.nan
        
        annual_return = clean_returns.mean() * periods_per_year
        dd_info = self._max_drawdown_series(returns, 1.0)
        max_dd = abs(dd_info['max_drawdown'])
        
        if max_dd == 0:
            return np.inf
        
        return (annual_return - risk_free_rate) / max_dd

    def run_full_risk_analysis(self,
                            market_data: Dict[str, Any],
                            position_value: float = 1000000,
                            confidence_level: float = 0.95,
                            risk_free_rate: float = 0.045,
                            n_simulations: int = 10000,
                            time_horizon: int = 252,
                            run_scenarios: bool = True,
                            random_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Run complete risk analytics on portfolio
        
        Parameters:
        -----------
        market_data : Output from DataPipeline.run_full_pipeline()
        pricing_results : Output from SyntheticPricer.run_full_pricing_analysis()
        statistical_results : Output from StatisticalModels.run_full_statistical_analysis()
        position_value : Position size per ticker (default: $1M)
        confidence_level : VaR/CVaR confidence (default: 95%)
        risk_free_rate : Risk-free rate (default: 4.5%)
        n_simulations : Monte Carlo simulations (default: 10,000)
        time_horizon : Days to simulate (default: 252)
        run_scenarios : Run scenario analysis (default: True)
        random_seed : Random seed for reproducibility
        
        Returns:
        --------
        Comprehensive risk report with:
        - VaR & CVaR
        - Sharpe & Sortino ratios
        - Max Drawdown
        - Monte Carlo projections
        - Scenario stress tests
        """
        
        print("\nRunning full risk analysis...")
        print(f"Position value: ${position_value:,.0f}, Confidence: {confidence_level:.0%}")
        
        # Extract data
        returns = market_data['returns']
        prices = market_data['prices']
        volatilities = market_data['volatility']
        tickers = list(returns.columns)

        print("Calculating VaR and CVaR...")
        
        var_results = {}
        cvar_results = {}
        
        for ticker in tickers:
            ticker_returns = returns[ticker]
            var_pct = self.calculate_var(ticker_returns, confidence_level, method='historical')
            cvar_pct = self.calculate_cvar(ticker_returns, confidence_level, method='historical')
            
            var_results[ticker] = var_pct * position_value
            cvar_results[ticker] = cvar_pct * position_value
        
        # Portfolio-level (equal weighted for simplicity)
        portfolio_returns = returns.mean(axis=1)
        portfolio_var = self.calculate_var(portfolio_returns, confidence_level) * position_value * len(tickers)
        portfolio_cvar = self.calculate_cvar(portfolio_returns, confidence_level) * position_value * len(tickers)

        print("Calculating risk-adjusted metrics...")
        
        sharpe_ratios = {}
        sortino_ratios = {}
        
        for ticker in tickers:
            ticker_returns = returns[ticker]
            sharpe_ratios[ticker] = self.calculate_sharpe_ratio(ticker_returns, risk_free_rate)
            sortino_ratios[ticker] = self.calculate_sortino_ratio(ticker_returns, risk_free_rate)
        
        # Portfolio-level
        portfolio_sharpe = self.calculate_sharpe_ratio(portfolio_returns, risk_free_rate)
        portfolio_sortino = self.calculate_sortino_ratio(portfolio_returns, risk_free_rate)

        print("Calculating drawdowns...")
        
        drawdown_results = {}
        
        for ticker in tickers:
            ticker_returns = returns[ticker]
            dd_info = self.calculate_max_drawdown(ticker_returns, initial_value=position_value)
            drawdown_results[ticker] = {
                'max_drawdown': dd_info['max_drawdown'],
                'max_drawdown_dollar': dd_info['max_drawdown_dollar'],
                'duration_days': dd_info['duration_days'],
                'recovered': dd_info['recovered']
            }
        
        # Portfolio-level
        portfolio_dd = self.calculate_max_drawdown(portfolio_returns, initial_value=position_value * len(tickers))
        
        print("Running Monte Carlo simulations...")
        
        mc_results = {}
        
        for ticker in tickers:
            ticker_returns = returns[ticker]
            mc = self.monte_carlo_simulation(
                returns=ticker_returns,
                initial_value=position_value,
                n_simulations=n_simulations,
                time_horizon=time_horizon,
                random_seed=random_seed
            )
            
            mc_results[ticker] = {
                'mean_final': mc['mean_final'],
                'median_final': mc['median_final'],
                'var_95': mc['var_95'],
                'cvar_95': mc['cvar_95'],
                'prob_loss': mc['prob_loss'],
                'prob_gain_10pct': mc['prob_gain_10pct'],
                'worst_case': mc['worst_case'],
                'best_case': mc['best_case']
            }
        
        # Portfolio-level Monte Carlo
        portfolio_mc = self.monte_carlo_simulation(
            returns=portfolio_returns,
            initial_value=position_value * len(tickers),
            n_simulations=n_simulations,
            time_horizon=time_horizon,
            random_seed=random_seed
        )
        
        scenario_results = {}
        
        if run_scenarios:
            print("Running scenario stress tests...")
            
            scenarios = self.create_standard_scenarios()
            
            for ticker in tickers:
                ticker_returns = returns[ticker]
                ticker_vol = volatilities[ticker].iloc[-1] if ticker in volatilities.columns else 0.25
                
                scenario_analysis = self.scenario_analysis(
                    returns=ticker_returns,
                    initial_value=position_value,
                    scenarios=scenarios,
                    n_simulations=n_simulations,
                    time_horizon=time_horizon,
                    current_volatility=ticker_vol,
                    random_seed=random_seed
                )
                
                # Simplify scenario results
                scenario_results[ticker] = {
                    scenario_name: {
                        'mean_final': results['mean_final'],
                        'mean_impact': results['mean_final'] - position_value,
                        'var_95': results['var_95'],
                        'prob_loss': results['prob_loss']
                    }
                    for scenario_name, results in scenario_analysis.items()
                }
            
            # Portfolio-level scenarios
            portfolio_scenario_analysis = self.scenario_analysis(
                returns=portfolio_returns,
                initial_value=position_value * len(tickers),
                scenarios=scenarios,
                n_simulations=n_simulations,
                time_horizon=time_horizon,
                random_seed=random_seed
            )
            
            scenario_results['portfolio'] = {
                scenario_name: {
                    'mean_final': results['mean_final'],
                    'mean_impact': results['mean_final'] - (position_value * len(tickers)),
                    'var_95': results['var_95'],
                    'prob_loss': results['prob_loss']
                }
                for scenario_name, results in portfolio_scenario_analysis.items()
            }
        
        risk_report = {
            # Summary (portfolio-level)
            'portfolio_summary': {
                'total_value': position_value * len(tickers),
                'num_positions': len(tickers),
                'var_95': portfolio_var,
                'cvar_95': portfolio_cvar,
                'sharpe_ratio': portfolio_sharpe,
                'sortino_ratio': portfolio_sortino,
                'max_drawdown': portfolio_dd['max_drawdown'],
                'max_drawdown_dollar': portfolio_dd['max_drawdown_dollar']
            },
            
            # Monte Carlo projections
            'monte_carlo': {
                'portfolio': {
                    'mean_final': portfolio_mc['mean_final'],
                    'median_final': portfolio_mc['median_final'],
                    'worst_5pct': portfolio_mc['var_95'] + (position_value * len(tickers)),
                    'prob_loss': portfolio_mc['prob_loss'],
                    'prob_gain_10pct': portfolio_mc['prob_gain_10pct']
                },
                'by_ticker': mc_results
            },
            
            # Risk metrics by ticker
            'by_ticker': {
                ticker: {
                    'var_95': var_results[ticker],
                    'cvar_95': cvar_results[ticker],
                    'sharpe_ratio': sharpe_ratios[ticker],
                    'sortino_ratio': sortino_ratios[ticker],
                    'max_drawdown': drawdown_results[ticker]['max_drawdown'],
                    'max_drawdown_dollar': drawdown_results[ticker]['max_drawdown_dollar'],
                    'drawdown_duration': drawdown_results[ticker]['duration_days']
                }
                for ticker in tickers
            },
            
            # Scenario stress tests
            'scenarios': scenario_results if run_scenarios else {},
            
            # Metadata
            'metadata': {
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'position_value_per_ticker': position_value,
                'confidence_level': confidence_level,
                'risk_free_rate': risk_free_rate,
                'n_simulations': n_simulations,
                'time_horizon': time_horizon,
                'tickers': tickers
            }
        }
        
        print("\nRISK ANALYSIS COMPLETE")
        
        print(f"\nPortfolio Summary (${risk_report['portfolio_summary']['total_value']:,.0f}):")
        print(f"  VaR(95%): ${abs(portfolio_var):,.0f}")
        print(f"  CVaR(95%): ${abs(portfolio_cvar):,.0f}")
        print(f"  Sharpe Ratio: {portfolio_sharpe:.2f}")
        print(f"  Sortino Ratio: {portfolio_sortino:.2f}")
        print(f"  Max Drawdown: {portfolio_dd['max_drawdown']:.2%}")
        
        print(f"\nMonte Carlo Projection ({time_horizon} days):")
        print(f"  Expected value: ${portfolio_mc['mean_final']:,.0f}")
        print(f"  Worst 5%: ${portfolio_mc['var_95'] + (position_value * len(tickers)):,.0f}")
        print(f"  Probability of loss: {portfolio_mc['prob_loss']:.1%}")
        
        if run_scenarios:
            print(f"\nWorst Scenario Impacts (Portfolio):")
            portfolio_scenarios = scenario_results.get('portfolio', {})
            worst_scenarios = sorted(
                portfolio_scenarios.items(),
                key=lambda x: x[1]['mean_impact']
            )[:3]
            
            for scenario_name, results in worst_scenarios:
                print(f"  {scenario_name}: ${results['mean_impact']:+,.0f} ({results['prob_loss']:.0%} loss prob)")
        
        print(f"\nRiskiest Tickers (by VaR):")
        worst_tickers = sorted(var_results.items(), key=lambda x: x[1])[:3]
        for ticker, var in worst_tickers:
            print(f"  {ticker}: ${abs(var):,.0f} VaR, {sharpe_ratios[ticker]:.2f} Sharpe")
        
        print("="*60 + "\n")
        
        return risk_report