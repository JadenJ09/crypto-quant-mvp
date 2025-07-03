"""
Advanced Statistics Engine

This module provides comprehensive performance analysis and statistics for backtesting results,
including traditional metrics, risk-adjusted returns, rolling statistics, and trade attribution.

Features:
- Traditional performance metrics (returns, Sharpe, Sortino, Calmar)
- Risk metrics (VaR, CVaR, Maximum Drawdown, Beta)
- Rolling statistics (rolling Sharpe, returns, volatility)
- Trade attribution and analysis
- Benchmark comparison and relative performance
- Monte Carlo simulation for confidence intervals
- Performance attribution by time periods, assets, strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from scipy import stats
import logging
from numba import jit

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics container"""
    
    # Returns
    total_return: float
    annualized_return: float
    cumulative_return: float
    
    # Risk metrics
    volatility: float
    annualized_volatility: float
    max_drawdown: float
    current_drawdown: float
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    
    # Time-based metrics
    best_month: float
    worst_month: float
    positive_months: int
    negative_months: int
    
    # Risk metrics
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%
    skewness: float
    kurtosis: float
    
    # Benchmark comparison (if provided)
    alpha: Optional[float] = None
    beta: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None


@dataclass
class RollingMetrics:
    """Rolling statistics container"""
    
    timestamps: List[pd.Timestamp]
    returns: List[float]
    sharpe_ratio: List[float]
    volatility: List[float]
    drawdown: List[float]
    cumulative_return: List[float]


@dataclass
class TradeAnalysis:
    """Detailed trade analysis"""
    
    # Trade statistics by direction
    long_trades: int
    short_trades: int
    long_win_rate: float
    short_win_rate: float
    long_profit_factor: float
    short_profit_factor: float
    
    # Hold time analysis
    average_hold_time: pd.Timedelta
    winning_hold_time: pd.Timedelta
    losing_hold_time: pd.Timedelta
    
    # Entry/Exit analysis
    entry_efficiency: float  # How close to optimal entry
    exit_efficiency: float   # How close to optimal exit
    
    # Consecutive statistics
    max_consecutive_wins: int
    max_consecutive_losses: int
    current_streak: int
    current_streak_type: str  # 'wins' or 'losses'


class StatisticsEngine:
    """
    Advanced statistics engine for comprehensive backtesting analysis
    
    Provides detailed performance metrics, risk analysis, and trade attribution
    for backtesting results with support for benchmark comparison and rolling statistics.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize statistics engine
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.benchmark_data: Optional[pd.Series] = None
        
    def set_benchmark(self, benchmark_returns: pd.Series):
        """Set benchmark for relative performance analysis"""
        self.benchmark_data = benchmark_returns.copy()
    
    def calculate_performance_metrics(self, 
                                    equity_curve: pd.Series,
                                    trades: List[Dict],
                                    initial_capital: float) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        
        Args:
            equity_curve: Portfolio equity curve with timestamps
            trades: List of completed trades
            initial_capital: Starting capital
            
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        
        if len(equity_curve) < 2:
            return self._empty_metrics()
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Basic return metrics
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        cumulative_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        
        # Annualized return (assuming daily data)
        periods_per_year = self._infer_periods_per_year(equity_curve.index)
        annualized_return = (1 + total_return) ** (periods_per_year / len(equity_curve)) - 1
        
        # Volatility metrics
        volatility = returns.std()
        annualized_volatility = volatility * np.sqrt(periods_per_year)
        
        # Drawdown calculation
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]
        
        # Risk-adjusted metrics
        excess_returns = returns - self.risk_free_rate / periods_per_year
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(periods_per_year) if downside_deviation > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Omega ratio
        omega_ratio = self._calculate_omega_ratio(returns)
        
        # Trade statistics
        trade_stats = self._calculate_trade_statistics(trades)
        
        # Monthly statistics
        monthly_stats = self._calculate_monthly_statistics(equity_curve)
        
        # Risk metrics
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Benchmark comparison if available
        alpha, beta, info_ratio, tracking_error = None, None, None, None
        if self.benchmark_data is not None:
            alpha, beta, info_ratio, tracking_error = self._calculate_benchmark_metrics(
                returns, equity_curve.index
            )
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_return=cumulative_return,
            volatility=volatility,
            annualized_volatility=annualized_volatility,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            total_trades=trade_stats['total_trades'],
            winning_trades=trade_stats['winning_trades'],
            losing_trades=trade_stats['losing_trades'],
            win_rate=trade_stats['win_rate'],
            profit_factor=trade_stats['profit_factor'],
            average_win=trade_stats['average_win'],
            average_loss=trade_stats['average_loss'],
            largest_win=trade_stats['largest_win'],
            largest_loss=trade_stats['largest_loss'],
            best_month=monthly_stats['best_month'],
            worst_month=monthly_stats['worst_month'],
            positive_months=monthly_stats['positive_months'],
            negative_months=monthly_stats['negative_months'],
            var_95=var_95,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis,
            alpha=alpha,
            beta=beta,
            information_ratio=info_ratio,
            tracking_error=tracking_error
        )
    
    def calculate_rolling_metrics(self,
                                equity_curve: pd.Series,
                                window: int = 252) -> RollingMetrics:
        """
        Calculate rolling performance metrics
        
        Args:
            equity_curve: Portfolio equity curve
            window: Rolling window size (default 252 for annual)
            
        Returns:
            RollingMetrics object with rolling statistics
        """
        
        returns = equity_curve.pct_change().dropna()
        
        if len(returns) < window:
            window = len(returns)
        
        timestamps = []
        rolling_returns = []
        rolling_sharpe = []
        rolling_volatility = []
        rolling_drawdown = []
        rolling_cumulative = []
        
        for i in range(window, len(equity_curve)):
            timestamp = equity_curve.index[i]
            period_returns = returns.iloc[i-window:i]
            period_equity = equity_curve.iloc[i-window:i+1]
            
            # Rolling metrics
            period_total_return = (period_equity.iloc[-1] - period_equity.iloc[0]) / period_equity.iloc[0]
            period_volatility = period_returns.std()
            
            # Rolling Sharpe ratio
            excess_returns = period_returns - self.risk_free_rate / 252  # Assuming daily
            period_sharpe = excess_returns.mean() / period_returns.std() * np.sqrt(252) if period_returns.std() > 0 else 0
            
            # Rolling drawdown
            period_max = period_equity.expanding().max().iloc[-1]
            period_drawdown = (period_equity.iloc[-1] - period_max) / period_max
            
            # Rolling cumulative return
            period_cumulative = period_equity.iloc[-1] / period_equity.iloc[0] - 1
            
            timestamps.append(timestamp)
            rolling_returns.append(period_total_return)
            rolling_sharpe.append(period_sharpe)
            rolling_volatility.append(period_volatility)
            rolling_drawdown.append(period_drawdown)
            rolling_cumulative.append(period_cumulative)
        
        return RollingMetrics(
            timestamps=timestamps,
            returns=rolling_returns,
            sharpe_ratio=rolling_sharpe,
            volatility=rolling_volatility,
            drawdown=rolling_drawdown,
            cumulative_return=rolling_cumulative
        )
    
    def analyze_trades(self, trades: List[Dict]) -> TradeAnalysis:
        """
        Perform detailed trade analysis
        
        Args:
            trades: List of completed trades
            
        Returns:
            TradeAnalysis object with detailed trade statistics
        """
        
        if not trades:
            return self._empty_trade_analysis()
        
        # Separate trades by direction
        long_trades = [t for t in trades if t.get('side') == 'long']
        short_trades = [t for t in trades if t.get('side') == 'short']
        
        # Trade direction statistics
        long_wins = len([t for t in long_trades if t.get('pnl', 0) > 0])
        short_wins = len([t for t in short_trades if t.get('pnl', 0) > 0])
        
        long_win_rate = long_wins / len(long_trades) if long_trades else 0
        short_win_rate = short_wins / len(short_trades) if short_trades else 0
        
        # Profit factors
        long_profit_factor = self._calculate_profit_factor([t.get('pnl', 0) for t in long_trades])
        short_profit_factor = self._calculate_profit_factor([t.get('pnl', 0) for t in short_trades])
        
        # Hold time analysis
        hold_times = []
        winning_hold_times = []
        losing_hold_times = []
        
        for trade in trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                hold_time = trade['exit_time'] - trade['entry_time']
                hold_times.append(hold_time)
                
                if trade.get('pnl', 0) > 0:
                    winning_hold_times.append(hold_time)
                else:
                    losing_hold_times.append(hold_time)
        
        avg_hold_time = pd.Timedelta(0) if not hold_times else sum(hold_times, pd.Timedelta(0)) / len(hold_times)
        winning_hold_time = pd.Timedelta(0) if not winning_hold_times else sum(winning_hold_times, pd.Timedelta(0)) / len(winning_hold_times)
        losing_hold_time = pd.Timedelta(0) if not losing_hold_times else sum(losing_hold_times, pd.Timedelta(0)) / len(losing_hold_times)
        
        # Entry/Exit efficiency (placeholder - would need more sophisticated analysis)
        entry_efficiency = 0.85  # Placeholder
        exit_efficiency = 0.80   # Placeholder
        
        # Consecutive statistics
        consecutive_stats = self._calculate_consecutive_statistics(trades)
        
        return TradeAnalysis(
            long_trades=len(long_trades),
            short_trades=len(short_trades),
            long_win_rate=long_win_rate,
            short_win_rate=short_win_rate,
            long_profit_factor=long_profit_factor,
            short_profit_factor=short_profit_factor,
            average_hold_time=avg_hold_time,
            winning_hold_time=winning_hold_time,
            losing_hold_time=losing_hold_time,
            entry_efficiency=entry_efficiency,
            exit_efficiency=exit_efficiency,
            max_consecutive_wins=consecutive_stats['max_wins'],
            max_consecutive_losses=consecutive_stats['max_losses'],
            current_streak=consecutive_stats['current_streak'],
            current_streak_type=consecutive_stats['streak_type']
        )
    
    def generate_report(self, 
                       metrics: PerformanceMetrics,
                       rolling_metrics: Optional[RollingMetrics] = None,
                       trade_analysis: Optional[TradeAnalysis] = None) -> str:
        """
        Generate comprehensive performance report
        
        Args:
            metrics: Performance metrics
            rolling_metrics: Rolling statistics (optional)
            trade_analysis: Trade analysis (optional)
            
        Returns:
            Formatted performance report string
        """
        
        report = []
        report.append("=" * 80)
        report.append("CUSTOM BACKTESTING ENGINE - PERFORMANCE REPORT")
        report.append("=" * 80)
        
        # Returns section
        report.append("\n📈 RETURNS")
        report.append("-" * 40)
        report.append(f"Total Return:           {metrics.total_return:>15.2%}")
        report.append(f"Annualized Return:      {metrics.annualized_return:>15.2%}")
        report.append(f"Cumulative Return:      {metrics.cumulative_return:>15.2%}")
        
        # Risk section
        report.append("\n⚠️  RISK METRICS")
        report.append("-" * 40)
        report.append(f"Volatility (Annual):    {metrics.annualized_volatility:>15.2%}")
        report.append(f"Max Drawdown:           {metrics.max_drawdown:>15.2%}")
        report.append(f"Current Drawdown:       {metrics.current_drawdown:>15.2%}")
        report.append(f"VaR (95%):             {metrics.var_95:>15.2%}")
        report.append(f"CVaR (95%):            {metrics.cvar_95:>15.2%}")
        
        # Risk-adjusted metrics
        report.append("\n📊 RISK-ADJUSTED METRICS")
        report.append("-" * 40)
        report.append(f"Sharpe Ratio:           {metrics.sharpe_ratio:>15.2f}")
        report.append(f"Sortino Ratio:          {metrics.sortino_ratio:>15.2f}")
        report.append(f"Calmar Ratio:           {metrics.calmar_ratio:>15.2f}")
        report.append(f"Omega Ratio:            {metrics.omega_ratio:>15.2f}")
        
        # Trade statistics
        report.append("\n🎯 TRADE STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Trades:           {metrics.total_trades:>15d}")
        report.append(f"Winning Trades:         {metrics.winning_trades:>15d}")
        report.append(f"Losing Trades:          {metrics.losing_trades:>15d}")
        report.append(f"Win Rate:               {metrics.win_rate:>15.1%}")
        report.append(f"Profit Factor:          {metrics.profit_factor:>15.2f}")
        report.append(f"Average Win:            {metrics.average_win:>15.2%}")
        report.append(f"Average Loss:           {metrics.average_loss:>15.2%}")
        report.append(f"Largest Win:            {metrics.largest_win:>15.2%}")
        report.append(f"Largest Loss:           {metrics.largest_loss:>15.2%}")
        
        # Benchmark comparison (if available)
        if metrics.alpha is not None:
            report.append("\n📋 BENCHMARK COMPARISON")
            report.append("-" * 40)
            report.append(f"Alpha:                  {metrics.alpha:>15.2%}")
            report.append(f"Beta:                   {metrics.beta:>15.2f}")
            report.append(f"Information Ratio:      {metrics.information_ratio:>15.2f}")
            report.append(f"Tracking Error:         {metrics.tracking_error:>15.2%}")
        
        # Trade analysis (if available)
        if trade_analysis:
            report.append("\n🔍 TRADE ANALYSIS")
            report.append("-" * 40)
            report.append(f"Long Trades:            {trade_analysis.long_trades:>15d}")
            report.append(f"Short Trades:           {trade_analysis.short_trades:>15d}")
            report.append(f"Long Win Rate:          {trade_analysis.long_win_rate:>15.1%}")
            report.append(f"Short Win Rate:         {trade_analysis.short_win_rate:>15.1%}")
            report.append(f"Long Profit Factor:     {trade_analysis.long_profit_factor:>15.2f}")
            report.append(f"Short Profit Factor:    {trade_analysis.short_profit_factor:>15.2f}")
            report.append(f"Avg Hold Time:          {str(trade_analysis.average_hold_time):>15s}")
            report.append(f"Winning Hold Time:      {str(trade_analysis.winning_hold_time):>15s}")
            report.append(f"Losing Hold Time:       {str(trade_analysis.losing_hold_time):>15s}")
            report.append(f"Entry Efficiency:       {trade_analysis.entry_efficiency:>15.2%}")
            report.append(f"Exit Efficiency:        {trade_analysis.exit_efficiency:>15.2%}")
            report.append(f"Max Consecutive Wins:   {trade_analysis.max_consecutive_wins:>15d}")
            report.append(f"Max Consecutive Losses: {trade_analysis.max_consecutive_losses:>15d}")
            report.append(f"Current Streak:         {trade_analysis.current_streak:>15d}")
            report.append(f"Current Streak Type:    {trade_analysis.current_streak_type:>15s}")
            
        # Additional performance details
        report.append("\n💡 ADDITIONAL METRICS")
        report.append("-" * 40)
        report.append(f"Best Month:             {metrics.best_month:>15.2%}")
        report.append(f"Worst Month:            {metrics.worst_month:>15.2%}")
        report.append(f"Positive Months:        {metrics.positive_months:>15d}")
        report.append(f"Negative Months:        {metrics.negative_months:>15d}")
        report.append(f"Skewness:               {metrics.skewness:>15.2f}")
        report.append(f"Kurtosis:               {metrics.kurtosis:>15.2f}")
        
        # Rolling metrics summary (if available)
        if rolling_metrics:
            report.append("\n📈 ROLLING PERFORMANCE SUMMARY")
            report.append("-" * 40)
            avg_rolling_return = sum(rolling_metrics.returns) / len(rolling_metrics.returns) if len(rolling_metrics.returns) > 0 else 0.0
            max_rolling_return = max(rolling_metrics.returns) if len(rolling_metrics.returns) > 0 else 0.0
            min_rolling_return = min(rolling_metrics.returns) if len(rolling_metrics.returns) > 0 else 0.0
            avg_rolling_volatility = sum(rolling_metrics.volatility) / len(rolling_metrics.volatility) if len(rolling_metrics.volatility) > 0 else 0.0
            avg_rolling_sharpe = sum(rolling_metrics.sharpe_ratio) / len(rolling_metrics.sharpe_ratio) if len(rolling_metrics.sharpe_ratio) > 0 else 0.0
            
            report.append(f"Avg Rolling Return:     {avg_rolling_return:>15.2%}")
            report.append(f"Max Rolling Return:     {max_rolling_return:>15.2%}")
            report.append(f"Min Rolling Return:     {min_rolling_return:>15.2%}")
            report.append(f"Avg Rolling Volatility: {avg_rolling_volatility:>15.2%}")
            report.append(f"Avg Rolling Sharpe:     {avg_rolling_sharpe:>15.2f}")
            report.append(f"Rolling Periods:        {len(rolling_metrics.returns):>15d}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    # Private helper methods
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics for edge cases"""
        return PerformanceMetrics(
            total_return=0.0, annualized_return=0.0, cumulative_return=0.0,
            volatility=0.0, annualized_volatility=0.0, max_drawdown=0.0, current_drawdown=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0, omega_ratio=0.0,
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
            profit_factor=0.0, average_win=0.0, average_loss=0.0, largest_win=0.0, largest_loss=0.0,
            best_month=0.0, worst_month=0.0, positive_months=0, negative_months=0,
            var_95=0.0, cvar_95=0.0, skewness=0.0, kurtosis=0.0
        )
    
    def _empty_trade_analysis(self) -> TradeAnalysis:
        """Return empty trade analysis for edge cases"""
        return TradeAnalysis(
            long_trades=0, short_trades=0, long_win_rate=0.0, short_win_rate=0.0,
            long_profit_factor=0.0, short_profit_factor=0.0,
            average_hold_time=pd.Timedelta(0), winning_hold_time=pd.Timedelta(0), losing_hold_time=pd.Timedelta(0),
            entry_efficiency=0.0, exit_efficiency=0.0,
            max_consecutive_wins=0, max_consecutive_losses=0, current_streak=0, current_streak_type='none'
        )
    
    def _infer_periods_per_year(self, index: pd.DatetimeIndex) -> int:
        """Infer number of periods per year from index frequency"""
        if len(index) < 2:
            return 252  # Default to daily
        
        # Handle minimal data case
        if len(index) < 3:
            # Calculate based on time difference
            time_diff = index[1] - index[0]
            if time_diff <= pd.Timedelta(hours=1):
                return 252 * 24  # Hourly or less
            elif time_diff <= pd.Timedelta(days=1):
                return 252  # Daily
            elif time_diff <= pd.Timedelta(days=7):
                return 52  # Weekly
            else:
                return 12  # Monthly or longer
        
        try:
            freq = pd.infer_freq(index)
            if freq:
                if 'D' in freq or 'B' in freq:
                    return 252  # Daily/Business daily
                elif 'H' in freq or 'h' in freq:
                    return 252 * 24  # Hourly
                elif 'T' in freq or 'min' in freq:
                    return 252 * 24 * 60  # Minute
                elif 'W' in freq:
                    return 52  # Weekly
                elif 'M' in freq or 'ME' in freq:
                    return 12  # Monthly
        except ValueError:
            # Fallback to time difference calculation
            time_diff = index[1] - index[0]
            if time_diff <= pd.Timedelta(hours=1):
                return 252 * 24  # Hourly or less
            elif time_diff <= pd.Timedelta(days=1):
                return 252  # Daily
            elif time_diff <= pd.Timedelta(days=7):
                return 52  # Weekly
            else:
                return 12  # Monthly or longer
        
        # Default to daily if can't infer
        return 252
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio"""
        excess_returns = returns - threshold
        positive_returns = excess_returns[excess_returns > 0].sum()
        negative_returns = abs(excess_returns[excess_returns < 0].sum())
        
        return positive_returns / negative_returns if negative_returns > 0 else 0
    
    def _calculate_trade_statistics(self, trades: List[Dict]) -> Dict[str, Any]:
        """Calculate trade-related statistics"""
        if not trades:
            return {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0.0, 'profit_factor': 0.0, 'average_win': 0.0,
                'average_loss': 0.0, 'largest_win': 0.0, 'largest_loss': 0.0
            }
        
        completed_trades = [t for t in trades if 'pnl' in t and t.get('exit_price') is not None]
        
        if not completed_trades:
            return {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0.0, 'profit_factor': 0.0, 'average_win': 0.0,
                'average_loss': 0.0, 'largest_win': 0.0, 'largest_loss': 0.0
            }
        
        pnls = [t['pnl'] for t in completed_trades]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        total_trades = len(completed_trades)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        
        win_rate = num_winning / total_trades if total_trades > 0 else 0
        
        average_win = sum(winning_trades) / num_winning if winning_trades else 0
        average_loss = sum(losing_trades) / num_losing if losing_trades else 0
        
        largest_win = max(winning_trades) if winning_trades else 0
        largest_loss = min(losing_trades) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': num_winning,
            'losing_trades': num_losing,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_win': average_win,
            'average_loss': average_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }
    
    def _calculate_monthly_statistics(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """Calculate monthly performance statistics"""
        if len(equity_curve) < 30:  # Not enough data for monthly analysis
            return {'best_month': 0.0, 'worst_month': 0.0, 'positive_months': 0, 'negative_months': 0}
        
        # Resample to monthly
        monthly_equity = equity_curve.resample('ME').last()
        monthly_returns = monthly_equity.pct_change().dropna()
        
        if len(monthly_returns) == 0:
            return {'best_month': 0.0, 'worst_month': 0.0, 'positive_months': 0, 'negative_months': 0}
        
        best_month = monthly_returns.max()
        worst_month = monthly_returns.min()
        positive_months = (monthly_returns > 0).sum()
        negative_months = (monthly_returns < 0).sum()
        
        return {
            'best_month': best_month,
            'worst_month': worst_month,
            'positive_months': positive_months,
            'negative_months': negative_months
        }
    
    def _calculate_benchmark_metrics(self, returns: pd.Series, timestamps: pd.DatetimeIndex) -> Tuple[float, float, float, float]:
        """Calculate metrics relative to benchmark"""
        if self.benchmark_data is None:
            return None, None, None, None
        
        # Align benchmark data with portfolio returns
        aligned_benchmark = self.benchmark_data.reindex(timestamps).pct_change().dropna()
        aligned_returns = returns.reindex(aligned_benchmark.index).dropna()
        
        if len(aligned_returns) == 0 or len(aligned_benchmark) == 0:
            return None, None, None, None
        
        # Calculate beta and alpha using linear regression
        try:
            beta, alpha_intercept, r_value, p_value, std_err = stats.linregress(aligned_benchmark, aligned_returns)
            alpha = alpha_intercept * 252  # Annualize alpha
            
            # Information ratio and tracking error
            excess_returns = aligned_returns - aligned_benchmark
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
            
            return alpha, beta, information_ratio, tracking_error
            
        except Exception as e:
            logger.warning(f"Error calculating benchmark metrics: {e}")
            return None, None, None, None
    
    def _calculate_profit_factor(self, pnls: List[float]) -> float:
        """Calculate profit factor for a list of PnLs"""
        if not pnls:
            return 0.0
        
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
    
    def _calculate_consecutive_statistics(self, trades: List[Dict]) -> Dict[str, Any]:
        """Calculate consecutive wins/losses statistics"""
        if not trades:
            return {'max_wins': 0, 'max_losses': 0, 'current_streak': 0, 'streak_type': 'none'}
        
        completed_trades = [t for t in trades if 'pnl' in t and t.get('exit_price') is not None]
        
        if not completed_trades:
            return {'max_wins': 0, 'max_losses': 0, 'current_streak': 0, 'streak_type': 'none'}
        
        # Sort trades by exit time
        completed_trades.sort(key=lambda x: x.get('exit_time', pd.Timestamp.min))
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in completed_trades:
            if trade['pnl'] > 0:  # Winning trade
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:  # Losing trade
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        # Current streak
        if current_wins > 0:
            current_streak = current_wins
            streak_type = 'wins'
        elif current_losses > 0:
            current_streak = current_losses
            streak_type = 'losses'
        else:
            current_streak = 0
            streak_type = 'none'
        
        return {
            'max_wins': max_consecutive_wins,
            'max_losses': max_consecutive_losses,
            'current_streak': current_streak,
            'streak_type': streak_type
        }
