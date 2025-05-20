import json
from typing import Dict, List, Any, Tuple, DefaultDict
import math
import statistics
import numpy as np
import jsonpickle
from collections import defaultdict, deque
import copy # Needed for deep copying if complex state causes issues

# Required imports from the provided datamodel
from datamodel import Listing, Order, OrderDepth, TradingState, Symbol, Trade, Observation, ConversionObservation

# Helper Functions (Unchanged unless noted)
def get_mid_price(order_depth: OrderDepth) -> float:
    if not order_depth.buy_orders or not order_depth.sell_orders:
        # Fallback: Use best bid/ask if only one side exists, or NaN if none
        if order_depth.buy_orders:
            return max(order_depth.buy_orders.keys())
        if order_depth.sell_orders:
            return min(order_depth.sell_orders.keys())
        return np.nan
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    return (best_bid + best_ask) / 2.0

def get_position(product: Symbol, state: TradingState) -> int:
    return state.position.get(product, 0)

# --- Trader Class with Adaptive Logic ---
class Trader:
    def __init__(self):
        """
        Initializes the Trader class, setting up position limits,
        parameters for adaptive strategies, and the persistent data structure.
        """
        self.POSITION_LIMITS = {
            'RAINFOREST_RESIN': 50, 'KELP': 50, 'SQUID_INK': 50,
            'CROISSANT': 250, 'JAM': 350, 'DJEMBE': 60,
            'PICNIC_BASKET1': 60, 'PICNIC_BASKET2': 100
            # Add other product limits if needed
        }

        # --- Adaptive Strategy Parameters ---
        # Note: These parameters require tuning and may impact performance/timeouts.
        self.max_history_len = 100 # Max length for indicator calculations

        # Regime Detection Parameters
        self.trend_ma_len = 50       # Lookback for trend moving average
        self.volatility_len = 20    # Lookback for volatility (std dev)
        self.trend_slope_threshold = 0.05 # Min abs slope for 'trending'
        self.volatility_threshold_ratio = 1.2 # Ratio to avg vol for 'high vol'

        # KELP (Mean Reversion) Parameters
        self.kelp_bollinger_len = 20
        self.kelp_base_std_multiplier = 2.0
        self.kelp_volatility_adapt_factor = 0.5 # Sensitivity of bands to vol
        self.kelp_base_order_size = 5

        # SQUID_INK (Momentum) Parameters
        self.squid_momentum_diff_len = 2 # Lookback for price change
        self.squid_base_threshold_std_ratio = 1.5 # Threshold ratio to vol
        self.squid_volatility_adapt_factor = 1.0 # Sensitivity of threshold to vol
        self.squid_base_order_size = 8

        # Dynamic Sizing Parameters
        self.size_volatility_adapt_factor = 0.6 # Sensitivity of size to vol
        self.min_order_size = 1

        # --- Persistent Data Store ---
        # Using defaultdict and deque for efficient history management
        self.persistent_data = {
            'price_history': defaultdict(lambda: deque(maxlen=self.max_history_len)),
            'avg_volatility': defaultdict(lambda: np.nan),
            'current_volatility': defaultdict(lambda: np.nan),
            'trend_ma': defaultdict(lambda: np.nan),
            'trend_slope': defaultdict(lambda: np.nan),
            'regime': defaultdict(lambda: 'Unknown')
        }
        # Note: PnL tracking removed for performance, can be added if needed
        # self.pnl_history = []

    # --- Indicator/Helper Calculation Methods ---

    def _update_indicators(self, product: Symbol):
        """
        Calculates volatility and trend indicators based on stored price history.
        NOTE: This uses approximations (mid-price std dev for vol, MA slope for trend)
              as full HLC data isn't readily available per tick.
              Calculations here can be computationally intensive.
        """
        prices = list(self.persistent_data['price_history'][product])

        # Volatility (Standard Deviation of recent mid-price changes)
        # Calculation only if enough data points are available
        if len(prices) >= self.volatility_len + 1:
            price_changes = np.diff(prices[-(self.volatility_len + 1):])
            current_vol = np.std(price_changes)
            self.persistent_data['current_volatility'][product] = current_vol

            # Update long-term average volatility (EMA approximation)
            avg_vol = self.persistent_data['avg_volatility'][product]
            if np.isnan(avg_vol):
                self.persistent_data['avg_volatility'][product] = current_vol
            else:
                alpha = 2 / (self.trend_ma_len + 1) # Smoother average
                self.persistent_data['avg_volatility'][product] = alpha * current_vol + (1 - alpha) * avg_vol
        else:
             # Not enough data, set volatility to NaN
             self.persistent_data['current_volatility'][product] = np.nan

        # Trend (Moving Average Slope)
        # Calculation only if enough data points are available
        if len(prices) >= self.trend_ma_len:
            current_ma = statistics.mean(prices[-self.trend_ma_len:])
            self.persistent_data['trend_ma'][product] = current_ma
            # Calculate slope using MA value from 5 periods ago for stability
            if len(prices) >= self.trend_ma_len + 5:
                 prev_ma = statistics.mean(prices[-(self.trend_ma_len + 5):-5])
                 # Avoid division by zero if prev_ma hasn't updated yet
                 slope = (current_ma - prev_ma) / 5 if prev_ma is not None else 0
                 self.persistent_data['trend_slope'][product] = slope
            else:
                 self.persistent_data['trend_slope'][product] = 0 # Flat slope if not enough history
        else:
            # Not enough data, set trend indicators to NaN or default
            self.persistent_data['trend_ma'][product] = np.nan
            self.persistent_data['trend_slope'][product] = np.nan


    def _get_regime(self, product: Symbol) -> str:
        """Determines market regime based on calculated trend slope and volatility."""
        slope = self.persistent_data['trend_slope'][product]
        current_vol = self.persistent_data['current_volatility'][product]
        avg_vol = self.persistent_data['avg_volatility'][product]

        # Return 'Unknown' if any indicator is not available
        if np.isnan(slope) or np.isnan(current_vol) or np.isnan(avg_vol):
            return 'Unknown'

        is_trending = abs(slope) > self.trend_slope_threshold
        is_high_vol = current_vol > avg_vol * self.volatility_threshold_ratio if avg_vol > 0 else False

        if is_trending:
            return 'Trending Up' if slope > 0 else 'Trending Down'
        else:
            return 'Ranging High Vol' if is_high_vol else 'Ranging Low Vol'

    def _calculate_dynamic_size(self, product: Symbol, base_size: int) -> int:
        """Calculates adaptive order size based on volatility ratio."""
        current_vol = self.persistent_data['current_volatility'][product]
        avg_vol = self.persistent_data['avg_volatility'][product]

        # Default to base size if volatility data is missing or zero
        if np.isnan(current_vol) or np.isnan(avg_vol) or avg_vol <= 0 or current_vol <= 0:
            return max(self.min_order_size, base_size)

        # Calculate size factor inversely proportional to vol, adjusted by sensitivity factor
        vol_factor = avg_vol / current_vol
        adaptive_factor = 1 + (vol_factor - 1) * self.size_volatility_adapt_factor
        # Clamp factor to avoid extreme sizes (e.g., 0.1x to 2x base size)
        adaptive_factor = max(0.1, min(adaptive_factor, 2.0))

        dynamic_size = int(round(base_size * adaptive_factor))
        # Ensure size is at least the minimum allowed
        return max(self.min_order_size, dynamic_size)


    # --- Main Trading Logic ---

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        """
        Main trading logic function called each time step.
        Updates indicators, determines regimes, calculates adaptive parameters,
        and places orders based on strategy rules.
        Handles persistent state loading and saving.
        Potential Optimization Point: This function's runtime can cause timeouts.
        """
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0

        # Load persistent data safely
        # This can be slow with complex data structures
        if state.traderData:
            try:
                loaded_data = jsonpickle.decode(state.traderData)
                # Carefully update persistent data, handling deques correctly
                for key, value_dict in loaded_data.items():
                    if key in self.persistent_data and isinstance(self.persistent_data[key], defaultdict):
                        target_dict = self.persistent_data[key]
                        for product, value in value_dict.items():
                            # Check if the target is a deque and reconstruct it
                            if isinstance(target_dict[product], deque) and isinstance(value, list):
                                target_dict[product] = deque(value, maxlen=target_dict.maxlen)
                            else:
                                target_dict[product] = value # Assign other types directly
            except Exception as e:
                print(f"Error loading traderData: {e}. State reset might occur.")
                # Consider resetting state fully if loading fails


        # --- Phase 1: Update Price History & Indicators ---
        # Loop through all products to update their data consistently
        for product in state.listings:
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                mid_price = get_mid_price(order_depth)

                if not np.isnan(mid_price):
                    # Store price
                    self.persistent_data['price_history'][product].append(mid_price)
                    # Update indicators (volatility, trend) for this product
                    self._update_indicators(product)
                    # Determine and store the current regime
                    self.persistent_data['regime'][product] = self._get_regime(product)


        # --- Phase 2: Execute Trading Logic ---
        # Loop through products again to make trading decisions
        for product in state.listings:
            # Skip if no order depth data is available for the product
            if product not in state.order_depths:
                continue

            order_depth = state.order_depths[product]
            orders: List[Order] = [] # Orders for the current product
            position = get_position(product, state)
            limit = self.POSITION_LIMITS.get(product, 0)
            current_regime = self.persistent_data['regime'].get(product, 'Unknown')
            mid_price = get_mid_price(order_depth) # Get current mid-price
            current_vol = self.persistent_data['current_volatility'].get(product, np.nan)

            # Skip trading this product if critical data is missing
            if np.isnan(mid_price) or np.isnan(current_vol) or current_regime == 'Unknown':
                continue

            # --- KELP Strategy Logic (Mean Reversion) ---
            if product == 'KELP':
                # Trade Condition: Only active in 'Ranging' regimes
                if 'Ranging' in current_regime:
                    prices = list(self.persistent_data['price_history'][product])
                    # Ensure enough data for Bollinger Band calculation
                    if len(prices) >= self.kelp_bollinger_len:
                        mean = statistics.mean(prices[-self.kelp_bollinger_len:])
                        std_dev = statistics.stdev(prices[-self.kelp_bollinger_len:]) if len(prices) > 1 else 0

                        if std_dev > 0: # Avoid division by zero or meaningless bands if std_dev is 0
                            avg_vol = self.persistent_data['avg_volatility'][product]
                            # Adapt band multiplier based on current vs average volatility
                            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
                            adaptive_multiplier = self.kelp_base_std_multiplier * (1 + (vol_ratio - 1) * self.kelp_volatility_adapt_factor)
                            adaptive_multiplier = max(1.0, adaptive_multiplier) # Min multiplier of 1

                            upper_band = mean + std_dev * adaptive_multiplier
                            lower_band = mean - std_dev * adaptive_multiplier

                            # Calculate dynamic order size based on volatility
                            dynamic_size = self._calculate_dynamic_size(product, self.kelp_base_order_size)

                            # Buy Signal: Price below lower band
                            if mid_price < lower_band:
                                buy_qty = min(dynamic_size, limit - position) # Respect position limit
                                if buy_qty > 0:
                                    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else int(round(mid_price + 0.5)) # Place order at best ask or slightly above mid
                                    orders.append(Order(product, best_ask, buy_qty))

                            # Sell Signal: Price above upper band
                            elif mid_price > upper_band:
                                sell_qty = min(dynamic_size, limit + position) # Respect position limit
                                if sell_qty > 0:
                                    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else int(round(mid_price - 0.5)) # Place order at best bid or slightly below mid
                                    orders.append(Order(product, best_bid, -sell_qty))

            # --- SQUID_INK Strategy Logic (Momentum) ---
            elif product == 'SQUID_INK':
                # Trade Condition: Only active in 'Trending' regimes, following the trend
                prices = list(self.persistent_data['price_history'][product])
                # Ensure enough data for price difference calculation
                if len(prices) >= self.squid_momentum_diff_len:
                    diff = prices[-1] - prices[-self.squid_momentum_diff_len]

                    # Adapt momentum threshold based on current volatility
                    adaptive_threshold = self.squid_base_threshold_std_ratio * current_vol

                    # Calculate dynamic order size
                    dynamic_size = self._calculate_dynamic_size(product, self.squid_base_order_size)

                    # Buy Signal: Trending Up & sufficient positive price difference
                    if current_regime == 'Trending Up' and diff > adaptive_threshold:
                        buy_qty = min(dynamic_size, limit - position)
                        if buy_qty > 0:
                             best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else int(round(mid_price + 0.5))
                             orders.append(Order(product, best_ask, buy_qty))

                    # Sell Signal: Trending Down & sufficient negative price difference
                    elif current_regime == 'Trending Down' and diff < -adaptive_threshold:
                        sell_qty = min(dynamic_size, limit + position)
                        if sell_qty > 0:
                            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else int(round(mid_price - 0.5))
                            orders.append(Order(product, best_bid, -sell_qty))

            # Add other product-specific strategies here...

            # Add generated orders for the current product to the result dictionary
            if orders:
                result[product] = orders

        # --- Phase 3: Save Persistent Data ---
        # Serialize the state for the next iteration
        # This can also be slow; ensure data structure is efficient
        serializable_data = {}
        # Convert deques to lists for JSON compatibility
        for key, value_dict in self.persistent_data.items():
            serializable_data[key] = {}
            if isinstance(value_dict, defaultdict):
                for product, value in value_dict.items():
                    if isinstance(value, deque):
                        serializable_data[key][product] = list(value)
                    else:
                         serializable_data[key][product] = value

        # Use jsonpickle without unpicklable=False if state needs to be fully restored
        # Using unpicklable=False makes it pure JSON but loses deque type info on load
        # Relying on the defaultdict structure in __init__ to restore types
        try:
             trader_data = jsonpickle.encode(serializable_data)
        except Exception as e:
             print(f"Error encoding traderData: {e}")
             trader_data = "" # Send empty string if encoding fails


        # Return the orders, conversions, and serialized state
        return result, conversions, trader_data