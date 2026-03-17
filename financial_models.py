"""
Comparing three different ways to predict S&P 500 prices:
1. Online ARIMA (River) - Updates in real-time
2. Batch ARIMA - The standard way (refitting every now and then)
3. Ternary Classifier - Just predicts Up/Down/Flat

Based on the trading strategy from Amjad & Shah (2017).
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import deque
from typing import Optional, Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*convergence.*')
import logging
logging.getLogger('statsmodels').setLevel(logging.ERROR)

# River imports
try:
    from river import time_series, linear_model, preprocessing, compose, metrics
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    print("Warning: River not installed. Install with: pip install river")

# Statsmodels for batch ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Install with: pip install statsmodels")


# --- Model 1: Online ARIMA (River) ---

class OnlineSNARIMAX:
    """
    Streaming ARIMA model using River.
    It learns one by one and doesn't need full retraining.
    """
    
    def __init__(self, p: int = 2, d: int = 1, q: int = 2):
        # standard ARIMA params

        if not RIVER_AVAILABLE:
            raise ImportError("River required: pip install river")
        
        self.name = "Online SNARIMAX"
        self.p = p
        self.d = d
        self.q = q
        
        # Init River model
        # m=1 because daily data has no seasonality
        self.model = time_series.SNARIMAX(
            p=p, d=d, q=q,
            m=1,
            sp=0, sd=0, sq=0
        )
        
        self.prices_seen = 0
        self.last_price = None
        # Give it about 3 months of data before trusting it
        self.warmup_period = 60
    
    def learn_one(self, price: float) -> None:
        """Update model with new price."""
        self.model.learn_one(price)
        self.last_price = price
        self.prices_seen += 1
    
    def forecast(self, horizon: int = 1) -> Optional[float]:
        """Predict future price."""
        if self.prices_seen < self.warmup_period:
            return self.last_price if self.last_price else None
        
        try:
            forecasts = self.model.forecast(horizon=horizon)
            if forecasts and len(forecasts) > 0:
                return float(forecasts[-1])
        except Exception as e:
            pass
        
        return self.last_price
    
    def is_ready(self) -> bool:
        """Check if model has seen enough data."""
        return self.prices_seen >= self.warmup_period


# --- Model 2: Batch ARIMA (Statsmodels) ---

class BatchARIMABaseline:
    """
    Standard ARIMA that we refit periodically.
    Used as a baseline to compare against the online version.
    """
    
    def __init__(self, window_size: int = 252, refit_interval: int = 60,
                 order: Tuple[int, int, int] = (2, 1, 2)):
        # Default: use 1 year of data, refit every 2 months
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required: pip install statsmodels")
        
        self.name = "Batch ARIMA"
        self.window_size = window_size
        self.refit_interval = refit_interval
        self.order = order
        
        self.price_history = []
        self.model = None
        self.last_fit_idx = 0
        self.last_forecast = None
    
    def learn_one(self, price: float) -> None:
        """Store price, maybe refit."""
        self.price_history.append(price)
        
        # Refit if we have enough data and it's time
        n = len(self.price_history)
        if n >= self.window_size and (n - self.last_fit_idx) >= self.refit_interval:
            self._fit_model()
    
    def _fit_model(self) -> None:
        """Fit ARIMA on recent window."""
        try:
            window = self.price_history[-self.window_size:]
            self.model = ARIMA(window, order=self.order)
            self.model = self.model.fit()
            self.last_fit_idx = len(self.price_history)
        except Exception as e:
            print(f"[BatchARIMA] Fit failed: {e}")  # Log instead of silent pass
    
    def forecast(self, horizon: int = 1) -> Optional[float]:
        """Get prediction from current model."""
        if self.model is None:
            return self.price_history[-1] if self.price_history else None
        
        try:
            fc = self.model.forecast(steps=horizon)
            # Robust: handle both pandas Series and numpy array
            self.last_forecast = float(fc.iloc[-1] if hasattr(fc, "iloc") else fc[-1])
            return self.last_forecast
        except Exception as e:
            print(f"[BatchARIMA] Forecast failed: {e}")
            return self.price_history[-1] if self.price_history else None
    
    def is_ready(self) -> bool:
        """Check if model has been fitted."""
        return self.model is not None


# --- Model 3: Online Ternary Classifier (River) ---

class OnlineTernaryClassifier:
    """
    Predicts: Up (1), Down (-1), or Flat (0).
    
    Based on the Amjad & Shah paper. It uses a confidence threshold
    so it only bets when it's sure.
    """
    
    def __init__(self, history_length: int = 5, confidence_threshold: float = 0.55,
                 change_threshold: float = 0.005, horizon: int = 1):
        # history_length: how far back to look for patterns
        # confidence_threshold: only predict if prob > this
        # change_threshold: how much price moves to count as up/down
        if not RIVER_AVAILABLE:
            raise ImportError("River required: pip install river")
        
        self.name = "Ternary Classifier"
        self.history_length = history_length
        self.confidence_threshold = confidence_threshold
        self.change_threshold = change_threshold
        self.horizon = horizon
        
        # River online classifier pipeline
        # Use SoftmaxRegression for proper multiclass probabilities
        self.model = compose.Pipeline(
            ('scale', preprocessing.StandardScaler()),
            ('clf', linear_model.SoftmaxRegression())
        )
        
        # Rolling window of ternary states (for 1-step feature extraction)
        self.ternary_history = deque(maxlen=history_length + 1)
        self.price_history = deque(maxlen=max(history_length + 2, horizon + 2))
        
        # Queue for delayed training (when looking N days ahead)
        self.pending_labels = deque()
        
        # Tracking
        self.n_learned = 0
        self.n_predicted = 0
        self.classes_seen = set()
        self.current_t = 0
    
    def _compute_ternary_label(self, price_prev: float, price_curr: float) -> int:
        """
        Decide if price went Up (1), Down (-1), or stayed Flat (0).
        """
        change = price_curr - price_prev
        threshold = self.change_threshold * price_prev
        
        if change > threshold:
            return 1
        elif change < -threshold:
            return -1
        else:
            return 0
    
    def _extract_features(self) -> Optional[Dict[str, float]]:
        """
        Get features like:
        1. Last direction
        2. Counts of up/down/flat recently
        3. Longest run of same direction
        """
        if len(self.ternary_history) < self.history_length:
            return None
        
        recent = list(self.ternary_history)[-self.history_length:]
        
        # Feature 1: Latest direction
        latest_direction = float(recent[-1])
        
        # Feature 2: Tally counts
        count_up = sum(1 for x in recent if x == 1)
        count_flat = sum(1 for x in recent if x == 0)
        count_down = sum(1 for x in recent if x == -1)
        
        # Feature 3: Maximum consecutive run
        max_run = self._compute_max_run(recent)
        
        # Additional features for richer representation
        momentum = sum(recent)  # Net direction
        
        return {
            'latest_direction': latest_direction,
            'count_up': float(count_up),
            'count_flat': float(count_flat),
            'count_down': float(count_down),
            'max_run': float(max_run),
            'momentum': float(momentum)
        }
    
    def _compute_max_run(self, sequence: List[int]) -> int:
        """Compute maximum consecutive run length."""
        if not sequence:
            return 0
        
        max_run = 1
        current_run = 1
        
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        
        return max_run
    
    def learn_one(self, price: float) -> None:
        """
        Update model. handles delayed labels if we are predicting N days out.
        """
        self.price_history.append(price)
        self.current_t += 1
        
        if len(self.price_history) < 2:
            return
        
        # Compute 1-step ternary label for feature extraction (always needed)
        prev_price = self.price_history[-2]
        one_step_label = self._compute_ternary_label(prev_price, price)
        self.ternary_history.append(one_step_label)
        self.classes_seen.add(one_step_label)
        
        # Extract current features
        features = self._extract_features()
        
        if self.horizon == 1:
            # Immediate training for 1-step ahead
            if features is not None:
                self.model.learn_one(features, one_step_label)
                self.n_learned += 1
        else:
            # Delayed training for multi-step ahead
            # Store features and entry price for later labeling
            if features is not None:
                entry_price = self.price_history[-1]  # Current price
                self.pending_labels.append((features.copy(), entry_price, self.current_t))
            
            # Check if we can resolve any pending labels (labels from horizon steps ago)
            # STRICT CHECK: Only process item if it matches exactly the current horizon
            while self.pending_labels:
                stored_features, entry_price, stored_t = self.pending_labels[0]
                
                # If exact horizon match, we have the correct future price
                if self.current_t - stored_t == self.horizon:
                    delayed_label = self._compute_ternary_label(entry_price, price)
                    self.model.learn_one(stored_features, delayed_label)
                    self.classes_seen.add(delayed_label)
                    self.n_learned += 1
                    self.pending_labels.popleft()
                
                # If we somehow missed the window (shouldn't happen in this loop), discard
                elif self.current_t - stored_t > self.horizon:
                    self.pending_labels.popleft()
                
                # If not enough time yet, stop checking
                else: 
                    break
    
    def predict_one(self) -> Tuple[int, float]:
        """
        Predict next ternary direction with confidence.
        
        Returns
        -------
        prediction : int
            -1 (down), 0 (hold/no confidence), +1 (up)
        confidence : float
            Probability of predicted class
        """
        features = self._extract_features()
        
        if features is None:
            return 0, 0.0
        
        try:
            # Get probability distribution
            proba = self.model.predict_proba_one(features)
            
            if not proba:
                return 0, 0.0
            
            # Find most likely class
            best_class = max(proba, key=proba.get)
            confidence = proba[best_class]
            
            # Apply confidence threshold
            if confidence >= self.confidence_threshold:
                self.n_predicted += 1
                return int(best_class), confidence
            else:
                return 0, confidence  # Hold if low confidence
                
        except Exception:
            return 0, 0.0
    
    def is_ready(self) -> bool:
        """Check if model has learned enough."""
        return self.n_learned >= self.history_length + 5

###
# --- EVALUATION FRAMEWORK ---
###

def evaluate_models(data: pd.DataFrame, trade_freq: int = 60, 
                    approach: str = 'next_step',
                    arima_order: Tuple[int, int, int] = (2, 1, 2)) -> Dict:
    """
    Run all 3 models on the data.
    Prediction first, then learn (prequential).
    
    Parameters
    ----------
    arima_order : Tuple[int, int, int]
        (p, d, q) parameters for ARIMA models (both Online SNARIMAX and Batch ARIMA).
    """
    prices = data['Open'].values
    dates = data.index
    n = len(prices)
    
    p, d, q = arima_order
    
    # Approach A: next_step uses horizon=1
    # Approach B: multi_day uses horizon=trade_freq with non-overlapping trades
    if approach == 'next_step':
        horizon = 1
        trade_every = 1  # Trade every day
        periods_per_year = 252
    else:  # multi_day
        horizon = trade_freq
        trade_every = trade_freq  # Non-overlapping: trade every N days
        periods_per_year = 252 / trade_freq
    
    # Initialize models
    models = {
        'Online SNARIMAX': OnlineSNARIMAX(p=p, d=d, q=q),
        'Batch ARIMA': BatchARIMABaseline(window_size=252, refit_interval=60, order=arima_order),
        'Ternary Classifier': OnlineTernaryClassifier(
            history_length=5, 
            confidence_threshold=0.55,
            change_threshold=0.005,
            horizon=horizon  # Match training labels to evaluation horizon
        )
    }
    
    # Results storage
    results = {name: {
        'predictions': [],
        'actuals': [],  # For MAE/RMSE calculation
        'pnl': [],  # Trade-only P&L (when direction != 0)
        'pnl_dates': [],  # Dates of trades
        'pnl_opportunity': [],  # P&L for ALL opportunities (including 0 for flat)
        'pnl_opportunity_dates': [],  # All opportunity dates
        'trades': 0,
        'correct': 0,
        'total_signals': 0,
        'directional_correct': 0  # Separate from trading accuracy
    } for name in models}
    
    approach_desc = "next-step (daily)" if approach == 'next_step' else f"multi-day ({horizon}-day horizon)"
    print(f"\nEvaluating models on {n} trading days...")
    print(f"Approach: {approach_desc}")
    print(f"Trade frequency: every {trade_every} day(s)\n")
    
    # Track last trade time for non-overlapping trades
    last_trade_t = {name: -trade_every for name in models}
    
    # Get ternary classifier's change_threshold for unified label computation
    tc = models['Ternary Classifier']
    change_threshold = tc.change_threshold
    
    def compute_ternary_label(price_prev, price_curr, threshold):
        """Unified ternary label computation (same as classifier's training)."""
        change = price_curr - price_prev
        thresh_val = threshold * price_prev
        if change > thresh_val:
            return 1
        elif change < -thresh_val:
            return -1
        else:
            return 0
    
    
    # Store returns for Sharpe calculation
    # We want a full series (zeros included) to compare properly
    complete_returns = {name: [] for name in models}
    trade_opportunity_dates = []
    
    # PREQUENTIAL LOOP:
    # 1. Predict (using history up to t-1)
    # 2. See what happened, calc P&L
    # 3. Learn (add t to history)
    for t in range(n):
        current_price = prices[t]
        
        # Determine if this is a trade opportunity (for non-overlapping trades)
        is_trade_opportunity = (t % trade_every == 0) and (t + horizon < n)
        
        if is_trade_opportunity:
            trade_opportunity_dates.append(t)
            future_price = prices[t + horizon]
            actual_change = future_price - current_price
        
        for name, model in models.items():
            # --- PREDICT PHASE (before peeking at current price) ---
            if is_trade_opportunity and model.is_ready():
                if isinstance(model, OnlineTernaryClassifier):
                    # Use same logic as training to see what actually happened
                    actual_direction = compute_ternary_label(current_price, future_price, change_threshold)
                    
                    direction, confidence = model.predict_one()
                    results[name]['predictions'].append(direction)
                    results[name]['actuals'].append(actual_direction)
                    
                    # Compute P&L (0 if holding/not trading)
                    ratio = future_price / current_price
                    if direction != 0:
                        pnl = (ratio - 1.0) if direction == 1 else (1.0 - ratio)
                        results[name]['pnl'].append(pnl)
                        results[name]['pnl_dates'].append(t)
                        results[name]['trades'] += 1
                        
                        # Trade-conditional accuracy
                        if (direction == 1 and actual_direction == 1) or \
                           (direction == -1 and actual_direction == -1):
                            results[name]['correct'] += 1
                        results[name]['total_signals'] += 1
                        
                        complete_returns[name].append(pnl)
                        results[name]['pnl_opportunity'].append(pnl)
                    else:
                        # Flat position: 0 return for this opportunity
                        complete_returns[name].append(0.0)
                        results[name]['pnl_opportunity'].append(0.0)
                    
                    # Record opportunity date for ALL predictions
                    results[name]['pnl_opportunity_dates'].append(t)
                    
                    # Directional accuracy (including 0 predictions)
                    if direction == actual_direction:
                        results[name]['directional_correct'] += 1
                
                else:
                    # Regression models: forecast at the specified horizon
                    forecast = model.forecast(horizon=horizon)
                    results[name]['predictions'].append(forecast)
                    results[name]['actuals'].append(future_price)
                    
                    if forecast is not None:
                        # Calculate realized ratio for P&L
                        ratio = future_price / current_price
                        
                        # Calculate pct change to see if it exceeds threshold
                        # Use same threshold as ternary classifier for fair comparison
                        change_pct = (forecast - current_price) / current_price
                        
                        if change_pct > change_threshold: # Long
                            pnl = ratio - 1.0
                            pred_direction = 1
                        elif change_pct < -change_threshold: # Short
                            pnl = 1.0 - ratio
                            pred_direction = -1
                        else: # Flat (within dead zone)
                            pnl = 0.0
                            pred_direction = 0
                        
                        # Use same ternary labeling for actuals (fair comparison)
                        actual_direction = compute_ternary_label(current_price, future_price, change_threshold)
                        
                        results[name]['pnl'].append(pnl)
                        results[name]['pnl_dates'].append(t)
                        if pred_direction != 0:
                            results[name]['trades'] += 1
                        complete_returns[name].append(pnl)
                        
                        # Record for opportunity series (aligned across models)
                        results[name]['pnl_opportunity'].append(pnl)
                        results[name]['pnl_opportunity_dates'].append(t)
                        
                        # Trading accuracy (only if we traded)
                        if pred_direction != 0:
                            if pred_direction == actual_direction:
                                results[name]['correct'] += 1
                            results[name]['total_signals'] += 1
                        
                        # Directional accuracy
                        if pred_direction == actual_direction:
                            results[name]['directional_correct'] += 1
                    else:
                        # Failed forecast: flat for this opportunity
                        complete_returns[name].append(0.0)
                        results[name]['pnl_opportunity'].append(0.0)
                        results[name]['pnl_opportunity_dates'].append(t)
            
            elif is_trade_opportunity and not model.is_ready():
                # Model not ready: flat return
                complete_returns[name].append(0.0)
                results[name]['pnl_opportunity'].append(0.0)
                results[name]['pnl_opportunity_dates'].append(t)
            
            # --- LEARN PHASE (now we can show the model the current price) ---
            model.learn_one(current_price)
    
    # Calculate final metrics
    for name in results:
        r = results[name]
        
        # Use COMPLETE return series for Sharpe (includes 0s for flat periods)
        complete_pnl = np.array(complete_returns[name])
        trading_pnl = np.array(r['pnl'])
        
        r['accuracy'] = r['correct'] / r['total_signals'] if r['total_signals'] > 0 else 0
        r['total_pnl'] = np.sum(trading_pnl) if len(trading_pnl) > 0 else 0
        r['avg_pnl'] = np.mean(trading_pnl) if len(trading_pnl) > 0 else 0
        
        # Sharpe ratio on COMPLETE return series (comparable across models)
        if len(complete_pnl) > 1 and np.std(complete_pnl) > 0:
            r['sharpe'] = np.sqrt(periods_per_year) * np.mean(complete_pnl) / np.std(complete_pnl)
        else:
            r['sharpe'] = 0.0
        
        # Also compute trade-conditional Sharpe for reference
        if len(trading_pnl) > 1 and np.std(trading_pnl) > 0:
            r['sharpe_trading'] = np.sqrt(periods_per_year) * np.mean(trading_pnl) / np.std(trading_pnl)
        else:
            r['sharpe_trading'] = 0.0
        
        # Coverage
        n_opportunities = len(trade_opportunity_dates)
        r['coverage'] = r['trades'] / n_opportunities if n_opportunities > 0 else 0
        
        # Forecasting metrics (MAE for regression, directional for all)
        if r['predictions'] and r['actuals']:
            # Safe zip to ensure alignment even if None predictions exist
            # (though we appended None/val to both lists synchronously)
            valid_pairs = [(p, a) for p, a in zip(r['predictions'], r['actuals']) 
                          if isinstance(p, (int, float)) and isinstance(a, (int, float))]
            
            if valid_pairs:
                if not isinstance(r['predictions'][0], int) or abs(r['predictions'][0]) > 1:
                     # Only compute MAE for price predictions, not directions
                     preds_, acts_ = zip(*valid_pairs)
                     r['mae'] = np.mean(np.abs(np.array(preds_) - np.array(acts_)))
        
        r['dir_accuracy'] = r['directional_correct'] / len(r['predictions']) if r['predictions'] else 0
    
    return results

###
# --- VISUALIZATION ---
###

def plot_comparison(data: pd.DataFrame, results: Dict, 
                   save_path: str = "comparison_report.png") -> plt.Figure:
    """
    Make a nice 4-panel chart.
    1. Price
    2. P&L
    3. Metrics
    4. Table
    """
    fig = plt.figure(figsize=(14, 10))
    
    colors = {
        'Online SNARIMAX': '#2E86AB',
        'Batch ARIMA': '#A23B72',
        'Ternary Classifier': '#F18F01'
    }
    
    # 1. Price chart
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(data.index, data['Open'], 'k-', linewidth=1.5, label='SPY (S&P 500 ETF)')
    # Dynamic title based on actual data date range
    start_year = data.index[0].year
    end_year = data.index[-1].year
    ax1.set_title(f'SPY (S&P 500 ETF) ({start_year}-{end_year})', fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Cumulative P&L (compounded equity curve using opportunity series)
    ax2 = plt.subplot(2, 2, 2)
    for name, r in results.items():
        if r['pnl_opportunity'] and r['pnl_opportunity_dates']:
            # Use compounded returns: equity = prod(1 + r_i) - 1
            cumulative = np.cumprod(1 + np.array(r['pnl_opportunity'])) - 1
            # Convert date indices to actual dates for proper alignment
            trade_dates = [data.index[i] for i in r['pnl_opportunity_dates']]
            ax2.plot(trade_dates, cumulative, label=f"{name} (Sharpe: {r['sharpe']:.2f})", 
                    color=colors.get(name, 'gray'), linewidth=2)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.set_title('Cumulative P&L (Compounded)', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Return')
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. Metrics comparison
    ax3 = plt.subplot(2, 2, 3)
    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.25
    
    metrics = ['accuracy', 'sharpe', 'coverage']
    labels = ['Trade Acc', 'Sharpe', 'Coverage']  # Renamed for clarity
    
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        values = [results[n][metric] for n in names]
        ax3.bar(x + i*width, values, width, label=label)
    
    ax3.set_ylabel('Value')
    ax3.set_title('Performance Metrics', fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Summary table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    table_data = []
    for name in names:
        r = results[name]
        table_data.append([
            name,
            f"{r['accuracy']:.3f}",
            f"{r['sharpe']:.2f}",
            f"{r['coverage']:.2f}",
            f"{r['trades']}"
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=['Model', 'Trade Acc', 'Sharpe', 'Coverage', 'Trades'],
        cellLoc='center',
        loc='center',
        colWidths=[0.28, 0.15, 0.15, 0.15, 0.15]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    
    return fig


def plot_paper_figure(data: pd.DataFrame, results: Dict, 
                      save_dir: str = "results/paper") -> None:
    """
    Generate paper-quality figures:
    - Figure 1: Price + Cumulative Return (2-panel, clean)
    - Table export: Metrics as separate file
    
    Follows academic paper conventions:
    - Subfigure labels (a), (b)
    - Clear axis labels with units
    - Compounded returns explicitly noted
    """
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    colors = {
        'Online SNARIMAX': '#2E86AB',
        'Batch ARIMA': '#A23B72',
        'Ternary Classifier': '#F18F01'
    }
    
    start_year = data.index[0].year
    end_year = data.index[-1].year
    
    # --- FIGURE 1: Price + Cumulative Return (2-panel) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # (a) Price panel
    ax1 = axes[0]
    ax1.plot(data.index, data['Open'], 'k-', linewidth=1.5)
    ax1.set_title(f'(a) SPY Open Price ({start_year}–{end_year})', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Price (USD)')
    ax1.set_xlabel('Date')
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.grid(True, alpha=0.3)
    
    # (b) Cumulative return panel
    ax2 = axes[1]
    for name, r in results.items():
        if r['pnl_opportunity'] and r['pnl_opportunity_dates']:
            cumulative = np.cumprod(1 + np.array(r['pnl_opportunity'])) - 1
            trade_dates = [data.index[i] for i in r['pnl_opportunity_dates']]
            ax2.plot(trade_dates, cumulative, 
                    label=f"{name} (Sharpe: {r['sharpe']:.2f})", 
                    color=colors.get(name, 'gray'), linewidth=2)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.set_title('(b) Cumulative Return (Compounded)', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Return')
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"figure_price_returns_{start_year}_{end_year}.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(fig_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"\nPaper figure saved to: {fig_path}")
    
    # --- TABLE: Metrics as separate CSV ---
    names = list(results.keys())
    table_data = []
    for name in names:
        r = results[name]
        table_data.append({
            'Model': name,
            'Trade Acc.': f"{r['accuracy']:.3f}",
            'Sharpe': f"{r['sharpe']:.2f}",
            'Coverage': f"{r['coverage']:.2f}",
            'Trades': r['trades']
        })
    
    table_df = pd.DataFrame(table_data)
    csv_path = os.path.join(save_dir, f"table_metrics_{start_year}_{end_year}.csv")
    table_df.to_csv(csv_path, index=False)
    print(f"Metrics table saved to: {csv_path}")
    
    # --- OPTIONAL: Table as figure (for appendix) ---
    fig_table, ax_table = plt.subplots(figsize=(8, 2.5))
    ax_table.axis('off')
    
    cell_text = [[name, f"{results[name]['accuracy']:.3f}", 
                  f"{results[name]['sharpe']:.2f}",
                  f"{results[name]['coverage']:.2f}",
                  f"{results[name]['trades']}"] for name in names]
    
    table = ax_table.table(
        cellText=cell_text,
        colLabels=['Model', 'Trade Acc.', 'Sharpe', 'Coverage', 'Trades'],
        cellLoc='center',
        loc='center',
        colWidths=[0.30, 0.15, 0.15, 0.15, 0.12]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    table_fig_path = os.path.join(save_dir, f"table_metrics_{start_year}_{end_year}.pdf")
    plt.savefig(table_fig_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Table figure saved to: {table_fig_path}")
    
    plt.close('all')


###
# --- SCENARIO DEFINITIONS ---
###

SCENARIOS = {
    'crisis': {
        'name': '2008 Financial Crisis',
        'start': '2006-01-01',
        'end': '2010-12-31',
        'description': 'Period covering the 2008 financial crisis and recovery'
    },
    'tranquil': {
        'name': 'Tranquil Period',
        'start': '2015-01-01',
        'end': '2018-12-31',
        'description': 'Relatively stable market period with low volatility'
    },
    'covid': {
        'name': 'COVID-19 Period',
        'start': '2019-01-01',
        'end': '2022-12-31',
        'description': 'Period covering COVID-19 pandemic and recovery'
    }
}

###
# --- MAIN EXECUTION ---
###

def main(scenario: str = 'crisis', trade_freq: int = 60, approach: str = 'next_step',
         arima_order: Tuple[int, int, int] = (2, 1, 2)):
    """
    Main runner.
    Pick a scenario (crisis/tranquil/covid) and go.
    
    Parameters
    ----------
    arima_order : Tuple[int, int, int]
        (p, d, q) parameters for ARIMA models.
    """
    
    if scenario not in SCENARIOS:
        print(f"ERROR: Unknown scenario '{scenario}'")
        print(f"Available scenarios: {', '.join(SCENARIOS.keys())}")
        return
    
    sc = SCENARIOS[scenario]
    
    print("=" * 70)
    print("FINANCIAL TIME SERIES FORECASTING: THREE-MODEL COMPARISON")
    print("=" * 70)
    print(f"\nScenario: {sc['name']}")
    print(f"Period: {sc['start']} to {sc['end']}")
    print(f"Description: {sc['description']}")
    
    # Check dependencies
    if not RIVER_AVAILABLE:
        print("ERROR: River ML required. Install: pip install river")
        return
    if not STATSMODELS_AVAILABLE:
        print("ERROR: statsmodels required. Install: pip install statsmodels")
        return
    
    # 1. Download data
    print(f"\n[1] Downloading S&P 500 data ({sc['start'][:4]}-{sc['end'][:4]})...")
    ticker = yf.Ticker("SPY")
    data = ticker.history(start=sc['start'], end=sc['end'])
    print(f"    Data shape: {data.shape}")
    print(f"    Price range: ${data['Open'].min():.2f} - ${data['Open'].max():.2f}")
    
    # 2. Evaluate models
    actual_horizon = 1 if approach == 'next_step' else trade_freq
    print(f"\n[2] Running prequential evaluation (approach={approach}, horizon={actual_horizon})...")
    results = evaluate_models(data, trade_freq=trade_freq, approach=approach, arima_order=arima_order)
    
    # 3. Print results (include ARIMA order in header)
    print(f"\nARIMA order: ({arima_order[0]}, {arima_order[1]}, {arima_order[2]})")
    print("\n" + "=" * 70)
    print(f"RESULTS SUMMARY - {sc['name'].upper()}")
    print("=" * 70)
    
    print(f"\n{'Model':<22} {'Accuracy':<10} {'Sharpe':<10} {'Coverage':<10} {'Trades':<8}")
    print("-" * 70)
    
    for name, r in results.items():
        print(f"{name:<22} {r['accuracy']:<10.3f} {r['sharpe']:<10.2f} "
              f"{r['coverage']:<10.2f} {r['trades']:<8}")
    
    # 4. Generate visualization
    print("\n[3] Generating visualization...")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    save_path = os.path.join(results_dir, f"comparison_{approach}_{scenario}.png")
    fig = plot_comparison(data, results, save_path=save_path)
    
    # Also generate paper-quality figures (in results/paper/)
    paper_dir = os.path.join(results_dir, "paper")
    plot_paper_figure(data, results, save_dir=paper_dir)
    
    plt.show()
    
    # 5. Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe'])
    best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"\n✓ Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1]['sharpe']:.2f})")
    print(f"✓ Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.3f})")
    
    # Ternary classifier insights
    tc = results.get('Ternary Classifier', {})
    if tc:
        print(f"\n✓ Ternary Classifier made {tc['trades']} trades "
              f"(coverage: {tc['coverage']:.1%})")
        print(f"  - Confidence threshold filtered out {1-tc['coverage']:.1%} of signals")
    
    print("\n" + "=" * 70)
    
    return results


def run_all_scenarios(trade_freq: int = 60, approach: str = 'next_step',
                      arima_order: Tuple[int, int, int] = (2, 1, 2)):
    """Run everything and print a big summary."""
    
    print("\n" + "=" * 70)
    print(f"RUNNING ALL SCENARIOS (approach={approach}, ARIMA order={arima_order})")
    print("=" * 70)
    
    all_results = {}
    
    for scenario in SCENARIOS:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {SCENARIOS[scenario]['name'].upper()}")
        print(f"{'='*70}")
        all_results[scenario] = main(scenario, trade_freq, approach, arima_order)
    
    # Print combined summary
    print("\n" + "=" * 70)
    print("COMBINED SUMMARY - ALL SCENARIOS")
    print("=" * 70)
    
    print(f"\n{'Scenario':<12} {'Model':<20} {'Accuracy':<10} {'Sharpe':<10}")
    print("-" * 60)
    
    for scenario, results in all_results.items():
        if results:
            for name, r in results.items():
                print(f"{scenario:<12} {name:<20} {r['accuracy']:<10.3f} {r['sharpe']:<10.2f}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare 3 financial trading models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  crisis    2008 Crash
  tranquil  Calm period (2015-2018)
  covid     Recent crazy times (2019-2022)
  all       Run them all
  
Approaches:
  next_step  Daily predictions (standard)
  multi_day  Predict N days out, non-overlapping trades

Examples:
  python financial_models.py -a multi_day -f 20   # 20-day trades
  python financial_models.py -s all -a next_step  # all scenarios, daily
  python financial_models.py --arima-p 3 --arima-d 1 --arima-q 3  # custom ARIMA(3,1,3)
        """
    )
    
    parser.add_argument(
        '--scenario', '-s',
        type=str,
        default='crisis',
        choices=['crisis', 'tranquil', 'covid', 'all'],
        help='Time period scenario to analyze (default: crisis)'
    )
    
    parser.add_argument(
        '--frequency', '-f',
        type=int,
        default=60,
        help='Trade frequency in days for multi_day approach (default: 60)'
    )
    
    parser.add_argument(
        '--approach', '-a',
        type=str,
        default='next_step',
        choices=['next_step', 'multi_day'],
        help='Evaluation approach (default: next_step)'
    )
    
    parser.add_argument(
        '--arima-p',
        type=int,
        default=2,
        help='ARIMA p parameter (AR order) for both Batch and Online ARIMA (default: 2)'
    )
    
    parser.add_argument(
        '--arima-d',
        type=int,
        default=1,
        help='ARIMA d parameter (differencing order) for both Batch and Online ARIMA (default: 1)'
    )
    
    parser.add_argument(
        '--arima-q',
        type=int,
        default=2,
        help='ARIMA q parameter (MA order) for both Batch and Online ARIMA (default: 2)'
    )
    
    args = parser.parse_args()
    
    arima_order = (args.arima_p, args.arima_d, args.arima_q)
    
    if args.scenario == 'all':
        run_all_scenarios(args.frequency, args.approach, arima_order)
    else:
        main(args.scenario, args.frequency, args.approach, arima_order)
