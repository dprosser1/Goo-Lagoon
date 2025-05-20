import json
from typing import Dict, List, Any, Tuple, DefaultDict
import math
import statistics
import numpy as np
import jsonpickle
from collections import defaultdict, deque
import copy  # Needed for deep copying if complex state causes issues

# --- Imports from the provided datamodel ---
from datamodel import Listing, Order, OrderDepth, TradingState, Symbol, Trade, Observation, ConversionObservation

def get_mid_price(order_depth: OrderDepth) -> float:
    """
    Computes a simple mid-price from the best bid & best ask.
    If only one side of the market exists, uses the best known side as fallback.
    Returns np.nan if the market is empty.
    """
    if not order_depth.buy_orders and not order_depth.sell_orders:
        return np.nan
    if not order_depth.buy_orders:
        return min(order_depth.sell_orders.keys())
    if not order_depth.sell_orders:
        return max(order_depth.buy_orders.keys())
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    return (best_bid + best_ask) / 2.0

def get_position(product: Symbol, state: TradingState) -> int:
    """Returns the current signed position for the specified product."""
    return state.position.get(product, 0)

class Trader:
    def __init__(self):
        """
        Initializes the Trader class, setting up position limits,
        parameters for the example strategies, and a persistent data structure.
        """
        # --- Position Limits ---
        self.POSITION_LIMITS = {
            # Round 2 products
            'RAINFOREST_RESIN': 50, 
            'KELP': 50, 
            'SQUID_INK': 50,
            'CROISSANT': 250, 
            'JAM': 350, 
            'DJEMBE': 60,
            'PICNIC_BASKET1': 60, 
            'PICNIC_BASKET2': 100,
            # Round 3 new products:
            'VOLCANIC_ROCK': 400,
            'VOLCANIC_ROCK_VOUCHER_9500': 200,
            'VOLCANIC_ROCK_VOUCHER_9750': 200,
            'VOLCANIC_ROCK_VOUCHER_10000': 200,
            'VOLCANIC_ROCK_VOUCHER_10250': 200,
            'VOLCANIC_ROCK_VOUCHER_10500': 200
        }

        # Example placeholder for voucher data:
        # (Strike price & hypothetical premium or any other param you want to track.)
        self.VOLCANIC_VOUCHER_INFO = {
            'VOLCANIC_ROCK_VOUCHER_9500':  {'strike': 9500,  'premium': 100}, 
            'VOLCANIC_ROCK_VOUCHER_9750':  {'strike': 9750,  'premium': 90},
            'VOLCANIC_ROCK_VOUCHER_10000': {'strike': 10000, 'premium': 80},
            'VOLCANIC_ROCK_VOUCHER_10250': {'strike': 10250, 'premium': 70},
            'VOLCANIC_ROCK_VOUCHER_10500': {'strike': 10500, 'premium': 60},
        }

        # --- Adaptive Strategy Parameters ---
        self.max_history_len = 100  # Max length for storing mid-price history

        # Regime detection (trend & volatility)
        self.trend_ma_len = 50
        self.volatility_len = 20
        self.trend_slope_threshold = 0.05
        self.volatility_threshold_ratio = 1.2

        # KELP (mean reversion) parameters
        self.kelp_bollinger_len = 20
        self.kelp_base_std_multiplier = 2.0
        self.kelp_volatility_adapt_factor = 0.5
        self.kelp_base_order_size = 5

        # SQUID_INK (momentum) parameters
        self.squid_momentum_diff_len = 2
        self.squid_base_threshold_std_ratio = 1.5
        self.squid_volatility_adapt_factor = 1.0
        self.squid_base_order_size = 8

        # Dynamic sizing
        self.size_volatility_adapt_factor = 0.6
        self.min_order_size = 1

        # --- Persistent Data Store ---
        # price_history[symbol] => deque of recent mid-prices
        self.persistent_data = {
            'price_history': defaultdict(lambda: deque(maxlen=self.max_history_len)),
            'avg_volatility': defaultdict(lambda: np.nan),
            'current_volatility': defaultdict(lambda: np.nan),
            'trend_ma': defaultdict(lambda: np.nan),
            'trend_slope': defaultdict(lambda: np.nan),
            'regime': defaultdict(lambda: 'Unknown')
        }

    # --- Indicator Calculation ---
    def _update_indicators(self, product: Symbol):
        """
        Calculates volatility and trend indicators based on stored price history.
        Basic approach: standard deviation of mid-price changes for volatility,
        slope of short vs. longer average for trend.
        """
        prices = list(self.persistent_data['price_history'][product])

        # Compute short-term volatility
        if len(prices) >= self.volatility_len + 1:
            recent_slice = prices[-(self.volatility_len + 1):]
            price_changes = np.diff(recent_slice)
            current_vol = np.std(price_changes)
            self.persistent_data['current_volatility'][product] = current_vol

            # Update long-term average volatility (EMA approach)
            avg_vol = self.persistent_data['avg_volatility'][product]
            if np.isnan(avg_vol):
                self.persistent_data['avg_volatility'][product] = current_vol
            else:
                alpha = 2 / (self.trend_ma_len + 1)
                self.persistent_data['avg_volatility'][product] = (
                    alpha * current_vol + (1 - alpha) * avg_vol
                )
        else:
            self.persistent_data['current_volatility'][product] = np.nan

        # Compute a moving-average-based slope for trend detection
        if len(prices) >= self.trend_ma_len:
            current_ma = statistics.mean(prices[-self.trend_ma_len:])
            self.persistent_data['trend_ma'][product] = current_ma

            # Compare to MA from ~5 points ago
            if len(prices) >= self.trend_ma_len + 5:
                prev_ma = statistics.mean(prices[-(self.trend_ma_len + 5):-5])
                slope = (current_ma - prev_ma) / 5
                self.persistent_data['trend_slope'][product] = slope
            else:
                self.persistent_data['trend_slope'][product] = 0
        else:
            self.persistent_data['trend_ma'][product] = np.nan
            self.persistent_data['trend_slope'][product] = np.nan

    def _get_regime(self, product: Symbol) -> str:
        """Classifies the current regime based on slope & volatility vs. averages."""
        slope = self.persistent_data['trend_slope'][product]
        current_vol = self.persistent_data['current_volatility'][product]
        avg_vol = self.persistent_data['avg_volatility'][product]

        if np.isnan(slope) or np.isnan(current_vol) or np.isnan(avg_vol):
            return 'Unknown'

        is_trending = abs(slope) > self.trend_slope_threshold
        is_high_vol = current_vol > avg_vol * self.volatility_threshold_ratio if avg_vol > 0 else False

        if is_trending:
            return 'Trending Up' if slope > 0 else 'Trending Down'
        else:
            return 'Ranging High Vol' if is_high_vol else 'Ranging Low Vol'

    def _calculate_dynamic_size(self, product: Symbol, base_size: int) -> int:
        """Adjusts order size based on how current volatility compares to average."""
        current_vol = self.persistent_data['current_volatility'][product]
        avg_vol = self.persistent_data['avg_volatility'][product]

        if (np.isnan(current_vol) or np.isnan(avg_vol) or 
            avg_vol <= 0 or current_vol <= 0):
            return max(self.min_order_size, base_size)

        vol_factor = avg_vol / current_vol
        adaptive_factor = 1 + (vol_factor - 1) * self.size_volatility_adapt_factor
        # Clamp the factor to avoid going too extreme
        adaptive_factor = max(0.1, min(adaptive_factor, 2.0))
        dynamic_size = int(round(base_size * adaptive_factor))
        return max(self.min_order_size, dynamic_size)

    # --- Main Trading Logic ---
    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        """
        The main method called each time step by the simulation.
        Must return:
          1) A dictionary of lists of Orders (keyed by symbol).
          2) The number of conversions (int) done that turn (if any).
          3) A string with your serialized 'traderData' (to persist state across time).
        """
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0  # Example usage for cross-product conversions if any

        # --- Load persistent data from prior run (if any) ---
        if state.traderData:
            try:
                loaded_data = jsonpickle.decode(state.traderData)
                # Merge loaded_data into self.persistent_data
                for key, value_dict in loaded_data.items():
                    if key in self.persistent_data:
                        if isinstance(self.persistent_data[key], defaultdict):
                            # Rebuild each product’s data carefully
                            for prod, val in value_dict.items():
                                # If it's a list, convert back to a deque
                                if isinstance(self.persistent_data[key][prod], deque) and isinstance(val, list):
                                    self.persistent_data[key][prod] = deque(val, maxlen=self.max_history_len)
                                else:
                                    self.persistent_data[key][prod] = val
            except Exception as e:
                print(f"[Round3Trader] Error loading traderData: {e}.")
                # In worst case, we just continue with fresh state

        # --- Phase 1: Update price histories & indicators for each product ---
        for product, listing in state.listings.items():
            if product not in state.order_depths:
                continue
            order_depth = state.order_depths[product]
            mid_price = get_mid_price(order_depth)
            if not np.isnan(mid_price):
                self.persistent_data['price_history'][product].append(mid_price)
                self._update_indicators(product)
                self.persistent_data['regime'][product] = self._get_regime(product)

        # --- Phase 2: Generate Orders ---
        for product, listing in state.listings.items():
            if product not in state.order_depths:
                continue

            order_depth = state.order_depths[product]
            position = get_position(product, state)
            limit = self.POSITION_LIMITS.get(product, 0)
            mid_price = get_mid_price(order_depth)
            regime = self.persistent_data['regime'].get(product, 'Unknown')
            current_vol = self.persistent_data['current_volatility'].get(product, np.nan)

            orders: List[Order] = []

            # Skip if we don't have a valid mid-price or volatility measurement
            if np.isnan(mid_price) or np.isnan(current_vol) or regime == 'Unknown':
                # (For simpler logic, we skip. You could still place naive orders.)
                continue

            # --- 1) The Round 2 KELP Mean-Reversion Example ---
            if product == 'KELP':
                if 'Ranging' in regime:
                    prices = list(self.persistent_data['price_history'][product])
                    if len(prices) >= self.kelp_bollinger_len:
                        mean = statistics.mean(prices[-self.kelp_bollinger_len:])
                        std_dev = statistics.pstdev(prices[-self.kelp_bollinger_len:])
                        if std_dev > 0:
                            avg_vol = self.persistent_data['avg_volatility'][product]
                            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
                            multiplier = self.kelp_base_std_multiplier * (
                                1 + (vol_ratio - 1) * self.kelp_volatility_adapt_factor
                            )
                            multiplier = max(1.0, multiplier)  # Ensure not too small
                            upper_band = mean + std_dev * multiplier
                            lower_band = mean - std_dev * multiplier
                            size = self._calculate_dynamic_size(product, self.kelp_base_order_size)

                            if mid_price < lower_band:
                                buy_qty = min(size, limit - position)
                                if buy_qty > 0:
                                    best_ask = (min(order_depth.sell_orders.keys())
                                                if order_depth.sell_orders
                                                else int(round(mid_price + 0.5)))
                                    orders.append(Order(product, best_ask, buy_qty))
                            elif mid_price > upper_band:
                                sell_qty = min(size, limit + position)
                                if sell_qty > 0:
                                    best_bid = (max(order_depth.buy_orders.keys())
                                                if order_depth.buy_orders
                                                else int(round(mid_price - 0.5)))
                                    orders.append(Order(product, best_bid, -sell_qty))

            # --- 2) The Round 2 SQUID_INK Momentum Example ---
            elif product == 'SQUID_INK':
                if regime.startswith('Trending'):
                    prices = list(self.persistent_data['price_history'][product])
                    if len(prices) >= self.squid_momentum_diff_len:
                        diff = prices[-1] - prices[-self.squid_momentum_diff_len]
                        threshold = self.squid_base_threshold_std_ratio * current_vol
                        size = self._calculate_dynamic_size(product, self.squid_base_order_size)

                        if regime == 'Trending Up' and diff > threshold:
                            buy_qty = min(size, limit - position)
                            if buy_qty > 0:
                                best_ask = (min(order_depth.sell_orders.keys())
                                            if order_depth.sell_orders
                                            else int(round(mid_price + 0.5)))
                                orders.append(Order(product, best_ask, buy_qty))
                        elif regime == 'Trending Down' and diff < -threshold:
                            sell_qty = min(size, limit + position)
                            if sell_qty > 0:
                                best_bid = (max(order_depth.buy_orders.keys())
                                            if order_depth.buy_orders
                                            else int(round(mid_price - 0.5)))
                                orders.append(Order(product, best_bid, -sell_qty))

            # --- 3) New Round 3: Vouchers + VOLCANIC_ROCK (Placeholder Logic) ---
            elif product == 'VOLCANIC_ROCK':
                # Example: Suppose we do a naive mean reversion or keep it simple.
                # You could adapt a real strategy. Here, we'll do minimal logic:
                # If mid_price < some threshold, buy a small amount; else if above, sell.
                # This is purely a placeholder to illustrate how you might incorporate it.
                threshold_price = 10000  # placeholder guess
                size = 5
                if mid_price < threshold_price and position < limit:
                    buy_qty = min(size, limit - position)
                    best_ask = (min(order_depth.sell_orders.keys())
                                if order_depth.sell_orders
                                else int(round(mid_price + 0.5)))
                    orders.append(Order(product, best_ask, buy_qty))
                elif mid_price > threshold_price and position > -limit:
                    sell_qty = min(size, limit + position)
                    best_bid = (max(order_depth.buy_orders.keys())
                                if order_depth.buy_orders
                                else int(round(mid_price - 0.5)))
                    orders.append(Order(product, best_bid, -sell_qty))

            elif product in self.VOLCANIC_VOUCHER_INFO:
                # Each voucher has its own strike & premium in self.VOLCANIC_VOUCHER_INFO[product].
                # A simple approach is: if the underlying mid_price for 'VOLCANIC_ROCK'
                # is well above the voucher’s strike (and the voucher’s market ask is “cheap”),
                # we might want to buy the voucher. This is just a naive example.
                info = self.VOLCANIC_VOUCHER_INFO[product]
                fair_underlying = None
                # We can attempt to retrieve mid-price for the underlying:
                if 'VOLCANIC_ROCK' in state.order_depths:
                    fair_underlying = get_mid_price(state.order_depths['VOLCANIC_ROCK'])
                if fair_underlying is None or np.isnan(fair_underlying):
                    fair_underlying = 10000  # Fallback guess if we can't get actual

                # If we believe the underlying might end above strike + premium => voucher could be valuable
                # Very naive approach: if fair_underlying > (strike + premium), we buy a bit. Otherwise we skip/sell.
                if fair_underlying > (info['strike'] + info['premium']):
                    # We want to buy the voucher if it’s not too expensive
                    if order_depth.sell_orders:
                        best_ask = min(order_depth.sell_orders.keys())
                        # We define some “fair price” for the voucher (purely an example)
                        # For instance, (fair_underlying - strike) - premium / time-decay factor
                        # Simplify to: potential payoff = fair_underlying - strike
                        # If best_ask < potential_payoff, we buy
                        payoff_estimate = (fair_underlying - info['strike'])
                        # Compare the payoff_estimate to best_ask to see if it's a good buy
                        if payoff_estimate > best_ask:  
                            size = 3  # small
                            buy_qty = min(size, limit - position)
                            if buy_qty > 0:
                                orders.append(Order(product, best_ask, buy_qty))
                else:
                    # If we suspect voucher is overpriced, we could sell it. 
                    if order_depth.buy_orders:
                        best_bid = max(order_depth.buy_orders.keys())
                        # Check if best_bid is above our "true" fair value
                        payoff_estimate = (fair_underlying - info['strike'])
                        if best_bid > payoff_estimate:
                            size = 3
                            sell_qty = min(size, limit + position)
                            if sell_qty > 0:
                                orders.append(Order(product, best_bid, -sell_qty))

            # If we generated any orders for this product, store them
            if orders:
                result[product] = orders

        # --- Phase 3: Save persistent data for next run ---
        serializable_data = {}
        for key, value_dict in self.persistent_data.items():
            serializable_data[key] = {}
            if isinstance(value_dict, defaultdict):
                for prod, val in value_dict.items():
                    if isinstance(val, deque):
                        serializable_data[key][prod] = list(val)
                    else:
                        serializable_data[key][prod] = val

        try:
            trader_data = jsonpickle.encode(serializable_data)
        except Exception as e:
            print(f"[Round3Trader] Error encoding traderData: {e}")
            trader_data = ""

        return result, conversions, trader_data
