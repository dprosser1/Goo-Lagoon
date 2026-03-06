"""
round4trader.py
===============
IMC Prosperity 3 - Round 4 Trader
Team: Goo-Lagoon

Round 4 introduced MAGNIFICENT_MACARONS, a product whose fair value is influenced
by real-world observable factors: hours of sunlight, sugar prices, shipping costs,
export/import tariffs, and storage capacity. Macarons can be traded locally via a
standard order book OR externally via a conversion mechanism (cross-market arbitrage).

Strategy Summary:
  - Maintain all prior strategies from Rounds 1-3 (KELP mean-reversion, SQUID_INK
    momentum, Picnic Basket stat-arb, VOLCANIC_ROCK / Vouchers options-based logic).
  - MAGNIFICENT_MACARONS: cross-market arbitrage using conversions.
      * Import when: foreignAsk + importTariff + transportFees < local best bid
      * Export when: local best ask < foreignBid - exportTariff - transportFees
      * Sunlight regime: when sunlightIndex is below a threshold, accumulate long
        positions (lower production expected => higher local prices ahead).
"""

import json
import math
import statistics
import numpy as np
import jsonpickle
from collections import defaultdict, deque
from typing import Dict, List, Tuple

from datamodel import (
    Listing, Order, OrderDepth, TradingState, Symbol,
    Trade, Observation, ConversionObservation
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_mid_price(order_depth: OrderDepth) -> float:
    if not order_depth.buy_orders and not order_depth.sell_orders:
        return np.nan
    if not order_depth.buy_orders:
        return float(min(order_depth.sell_orders.keys()))
    if not order_depth.sell_orders:
        return float(max(order_depth.buy_orders.keys()))
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    return (best_bid + best_ask) / 2.0


def get_position(product: Symbol, state: TradingState) -> int:
    return state.position.get(product, 0)


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------

class Trader:
    def __init__(self):
        # ---- Position Limits ----
        self.POSITION_LIMITS = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANT": 250,
            "JAM": 350,
            "DJEMBE": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
            "MAGNIFICENT_MACARONS": 75,
        }

        # Conversion limit per timestep for macarons
        self.MACARON_CONVERSION_LIMIT = 10

        # ---- Picnic Basket Composition ----
        self.BASKET1_COMPONENTS = {"CROISSANT": 6, "JAM": 3, "DJEMBE": 1}
        self.BASKET2_COMPONENTS = {"CROISSANT": 4, "JAM": 2}
        self.BASKET_SPREAD_THRESHOLD = 20  # SeaShells, trigger arb when mispricing > this

        # ---- Volcanic Voucher Strikes ----
        self.VOLCANIC_VOUCHER_STRIKES = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500,
        }
        self.VOUCHER_DAYS_TO_EXPIRY = 5  # Approximation for Black-Scholes T

        # ---- KELP Mean-Reversion ----
        self.kelp_bollinger_len = 20
        self.kelp_std_multiplier = 2.0
        self.kelp_base_size = 8

        # ---- SQUID_INK Momentum ----
        self.squid_momentum_len = 3
        self.squid_threshold_multiplier = 1.5
        self.squid_base_size = 8

        # ---- Regime / Volatility Detection ----
        self.volatility_len = 20
        self.trend_ma_len = 50
        self.trend_slope_threshold = 0.05
        self.volatility_threshold_ratio = 1.2

        # ---- MAGNIFICENT_MACARONS Parameters ----
        self.macaron_sunlight_threshold = 2500.0  # Below => low-sunlight regime
        self.macaron_arb_min_profit = 1.0          # Minimum net profit per unit to trade
        self.macaron_accumulate_size = 5           # Units to accumulate in low-sunlight regime
        self.macaron_base_size = 5                  # Normal market-making order size

        # ---- Dynamic Sizing ----
        self.size_vol_adapt_factor = 0.6
        self.min_order_size = 1
        self.max_history_len = 100

        # ---- Persistent State ----
        self.persistent_data = {
            "price_history": defaultdict(lambda: deque(maxlen=self.max_history_len)),
            "avg_volatility": defaultdict(lambda: np.nan),
            "current_volatility": defaultdict(lambda: np.nan),
            "trend_ma": defaultdict(lambda: np.nan),
            "trend_slope": defaultdict(lambda: np.nan),
            "regime": defaultdict(lambda: "Unknown"),
        }

    # -----------------------------------------------------------------------
    # Indicator helpers
    # -----------------------------------------------------------------------

    def _update_indicators(self, product: Symbol):
        prices = list(self.persistent_data["price_history"][product])

        if len(prices) >= self.volatility_len + 1:
            changes = np.diff(prices[-(self.volatility_len + 1):])
            cur_vol = float(np.std(changes))
            self.persistent_data["current_volatility"][product] = cur_vol
            avg_vol = self.persistent_data["avg_volatility"][product]
            if np.isnan(avg_vol):
                self.persistent_data["avg_volatility"][product] = cur_vol
            else:
                alpha = 2.0 / (self.trend_ma_len + 1)
                self.persistent_data["avg_volatility"][product] = (
                    alpha * cur_vol + (1 - alpha) * avg_vol
                )
        else:
            self.persistent_data["current_volatility"][product] = np.nan

        if len(prices) >= self.trend_ma_len:
            cur_ma = statistics.mean(prices[-self.trend_ma_len:])
            self.persistent_data["trend_ma"][product] = cur_ma
            if len(prices) >= self.trend_ma_len + 5:
                prev_ma = statistics.mean(prices[-(self.trend_ma_len + 5):-5])
                self.persistent_data["trend_slope"][product] = (cur_ma - prev_ma) / 5.0
            else:
                self.persistent_data["trend_slope"][product] = 0.0
        else:
            self.persistent_data["trend_ma"][product] = np.nan
            self.persistent_data["trend_slope"][product] = np.nan

    def _get_regime(self, product: Symbol) -> str:
        slope = self.persistent_data["trend_slope"][product]
        cur_vol = self.persistent_data["current_volatility"][product]
        avg_vol = self.persistent_data["avg_volatility"][product]
        if np.isnan(slope) or np.isnan(cur_vol) or np.isnan(avg_vol):
            return "Unknown"
        trending = abs(slope) > self.trend_slope_threshold
        high_vol = cur_vol > avg_vol * self.volatility_threshold_ratio if avg_vol > 0 else False
        if trending:
            return "Trending Up" if slope > 0 else "Trending Down"
        return "Ranging High Vol" if high_vol else "Ranging Low Vol"

    def _dynamic_size(self, product: Symbol, base: int) -> int:
        cur_vol = self.persistent_data["current_volatility"][product]
        avg_vol = self.persistent_data["avg_volatility"][product]
        if np.isnan(cur_vol) or np.isnan(avg_vol) or avg_vol <= 0 or cur_vol <= 0:
            return max(self.min_order_size, base)
        vol_factor = avg_vol / cur_vol
        factor = max(0.1, min(1 + (vol_factor - 1) * self.size_vol_adapt_factor, 2.0))
        return max(self.min_order_size, int(round(base * factor)))

    # -----------------------------------------------------------------------
    # Black-Scholes helpers for vouchers
    # -----------------------------------------------------------------------

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def _bs_call_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """Black-Scholes call price; r=0 approximation."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return max(0.0, S - K)
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self._norm_cdf(d1) - K * self._norm_cdf(d2)

    def _implied_vol(self, price: float, S: float, K: float, T: float) -> float:
        """Newton-Raphson implied vol from observed option price."""
        intrinsic = max(0.0, S - K)
        if price <= intrinsic + 1e-6:
            return 0.01
        sigma = 0.3
        for _ in range(30):
            c = self._bs_call_price(S, K, T, sigma)
            if T <= 0:
                break
            d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
            vega = S * self._norm_cdf(d1) * math.sqrt(T) * math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
            if abs(vega) < 1e-8:
                break
            sigma -= (c - price) / vega
            sigma = max(0.001, min(sigma, 5.0))
        return sigma

    # -----------------------------------------------------------------------
    # Product-specific strategies
    # -----------------------------------------------------------------------

    def _trade_kelp(self, state: TradingState) -> List[Order]:
        product = "KELP"
        orders: List[Order] = []
        order_depth = state.order_depths.get(product)
        if order_depth is None:
            return orders
        position = get_position(product, state)
        limit = self.POSITION_LIMITS[product]
        mid_price = get_mid_price(order_depth)
        regime = self.persistent_data["regime"].get(product, "Unknown")
        cur_vol = self.persistent_data["current_volatility"].get(product, np.nan)
        prices = list(self.persistent_data["price_history"][product])

        if np.isnan(mid_price) or np.isnan(cur_vol) or regime == "Unknown":
            return orders

        if "Ranging" in regime and len(prices) >= self.kelp_bollinger_len:
            mean = statistics.mean(prices[-self.kelp_bollinger_len:])
            std = statistics.pstdev(prices[-self.kelp_bollinger_len:])
            if std > 0:
                avg_vol = self.persistent_data["avg_volatility"][product]
                vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
                multiplier = max(1.0, self.kelp_std_multiplier * (1 + (vol_ratio - 1) * 0.5))
                upper = mean + std * multiplier
                lower = mean - std * multiplier
                size = self._dynamic_size(product, self.kelp_base_size)
                if mid_price < lower:
                    qty = min(size, limit - position)
                    if qty > 0 and order_depth.sell_orders:
                        best_ask = min(order_depth.sell_orders.keys())
                        orders.append(Order(product, best_ask, qty))
                elif mid_price > upper:
                    qty = min(size, limit + position)
                    if qty > 0 and order_depth.buy_orders:
                        best_bid = max(order_depth.buy_orders.keys())
                        orders.append(Order(product, best_bid, -qty))
        return orders

    def _trade_squid(self, state: TradingState) -> List[Order]:
        product = "SQUID_INK"
        orders: List[Order] = []
        order_depth = state.order_depths.get(product)
        if order_depth is None:
            return orders
        position = get_position(product, state)
        limit = self.POSITION_LIMITS[product]
        mid_price = get_mid_price(order_depth)
        regime = self.persistent_data["regime"].get(product, "Unknown")
        cur_vol = self.persistent_data["current_volatility"].get(product, np.nan)
        prices = list(self.persistent_data["price_history"][product])

        if np.isnan(mid_price) or np.isnan(cur_vol) or regime == "Unknown":
            return orders

        if regime.startswith("Trending") and len(prices) >= self.squid_momentum_len:
            diff = prices[-1] - prices[-self.squid_momentum_len]
            threshold = self.squid_threshold_multiplier * cur_vol
            size = self._dynamic_size(product, self.squid_base_size)
            if regime == "Trending Up" and diff > threshold:
                qty = min(size, limit - position)
                if qty > 0 and order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    orders.append(Order(product, best_ask, qty))
            elif regime == "Trending Down" and diff < -threshold:
                qty = min(size, limit + position)
                if qty > 0 and order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    orders.append(Order(product, best_bid, -qty))
        return orders

    def _trade_baskets(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        """Index arbitrage on Picnic Basket 1 and Basket 2."""
        result: Dict[Symbol, List[Order]] = {}

        def best_bid(sym):
            od = state.order_depths.get(sym)
            return max(od.buy_orders.keys()) if od and od.buy_orders else None

        def best_ask(sym):
            od = state.order_depths.get(sym)
            return min(od.sell_orders.keys()) if od and od.sell_orders else None

        def fair_basket(components):
            total = 0.0
            for comp, qty in components.items():
                ask = best_ask(comp)
                bid = best_bid(comp)
                if ask is None or bid is None:
                    return None
                total += ((ask + bid) / 2.0) * qty
            return total

        for basket, components in [
            ("PICNIC_BASKET1", self.BASKET1_COMPONENTS),
            ("PICNIC_BASKET2", self.BASKET2_COMPONENTS),
        ]:
            od = state.order_depths.get(basket)
            if od is None:
                continue
            fair = fair_basket(components)
            if fair is None:
                continue
            position = get_position(basket, state)
            limit = self.POSITION_LIMITS[basket]
            orders: List[Order] = []

            b_ask = best_ask(basket)
            b_bid = best_bid(basket)

            # Basket is underpriced: buy basket
            if b_ask is not None and (fair - b_ask) > self.BASKET_SPREAD_THRESHOLD:
                qty = min(5, limit - position)
                if qty > 0:
                    orders.append(Order(basket, b_ask, qty))

            # Basket is overpriced: sell basket
            if b_bid is not None and (b_bid - fair) > self.BASKET_SPREAD_THRESHOLD:
                qty = min(5, limit + position)
                if qty > 0:
                    orders.append(Order(basket, b_bid, -qty))

            if orders:
                result[basket] = orders
        return result

    def _trade_volcanic(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        """Options-based logic for Volcanic Rock and its vouchers."""
        result: Dict[Symbol, List[Order]] = {}
        vr_depth = state.order_depths.get("VOLCANIC_ROCK")
        if vr_depth is None:
            return result
        S = get_mid_price(vr_depth)
        if np.isnan(S):
            return result

        T = self.VOUCHER_DAYS_TO_EXPIRY / 365.0
        iv_estimates = []

        for voucher, strike in self.VOLCANIC_VOUCHER_STRIKES.items():
            od = state.order_depths.get(voucher)
            if od is None:
                continue
            v_mid = get_mid_price(od)
            if np.isnan(v_mid):
                continue
            iv = self._implied_vol(v_mid, S, strike, T)
            iv_estimates.append(iv)

        if not iv_estimates:
            return result

        avg_iv = statistics.mean(iv_estimates)

        for voucher, strike in self.VOLCANIC_VOUCHER_STRIKES.items():
            od = state.order_depths.get(voucher)
            if od is None:
                continue
            v_mid = get_mid_price(od)
            if np.isnan(v_mid):
                continue
            fair_price = self._bs_call_price(S, strike, T, avg_iv)
            position = get_position(voucher, state)
            limit = self.POSITION_LIMITS[voucher]
            orders: List[Order] = []

            if od.sell_orders:
                best_ask = min(od.sell_orders.keys())
                if best_ask < fair_price * 0.97:
                    qty = min(3, limit - position)
                    if qty > 0:
                        orders.append(Order(voucher, best_ask, qty))

            if od.buy_orders:
                best_bid = max(od.buy_orders.keys())
                if best_bid > fair_price * 1.03:
                    qty = min(3, limit + position)
                    if qty > 0:
                        orders.append(Order(voucher, best_bid, -qty))

            if orders:
                result[voucher] = orders

        return result

    def _trade_macarons(
        self, state: TradingState
    ) -> Tuple[Dict[Symbol, List[Order]], int]:
        """
        Cross-market arbitrage for MAGNIFICENT_MACARONS.

        Returns (orders_dict, conversions).
        Conversions > 0 = import from foreign island.
        Conversions < 0 = export to foreign island.
        """
        product = "MAGNIFICENT_MACARONS"
        orders: List[Order] = []
        conversions = 0

        od = state.order_depths.get(product)
        if od is None:
            return {}, 0

        # Fetch conversion observation
        conv_obs: ConversionObservation = (
            state.observations.conversionObservations.get(product)
            if state.observations and state.observations.conversionObservations
            else None
        )
        if conv_obs is None:
            return {}, 0

        position = get_position(product, state)
        limit = self.POSITION_LIMITS[product]

        # Effective costs for cross-market transactions
        import_cost = conv_obs.askPrice + conv_obs.importTariff + conv_obs.transportFees
        export_revenue = conv_obs.bidPrice - conv_obs.exportTariff - conv_obs.transportFees

        local_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        local_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        sunlight = conv_obs.sunlightIndex
        low_sunlight = sunlight < self.macaron_sunlight_threshold

        print(
            f"[MACARONS] sunlight={sunlight:.1f}, import_cost={import_cost:.2f}, "
            f"export_rev={export_revenue:.2f}, local_bid={local_bid}, local_ask={local_ask}, "
            f"pos={position}"
        )

        # ---- Regime: Low Sunlight => Accumulate Long ----
        if low_sunlight:
            # Accumulate locally if possible; avoid exporting
            if local_ask is not None and position < limit:
                qty = min(self.macaron_accumulate_size, limit - position)
                if qty > 0:
                    orders.append(Order(product, local_ask, qty))
            # Also import if cheaper than local ask
            if local_ask is not None and import_cost < local_ask:
                import_qty = min(
                    self.MACARON_CONVERSION_LIMIT,
                    limit - position - (
                        sum(
                            v for v in od.sell_orders.values()
                            if od.sell_orders
                        )
                        if orders
                        else 0
                    ),
                )
                import_qty = max(0, import_qty)
                conversions += import_qty

        else:
            # ---- Normal Regime: Two-Way Arbitrage ----

            # Arb: import from foreign, sell locally
            if local_bid is not None and (local_bid - import_cost) > self.macaron_arb_min_profit:
                import_qty = min(
                    self.MACARON_CONVERSION_LIMIT,
                    limit - position,
                )
                if import_qty > 0:
                    conversions += import_qty
                    # Immediately offer the imported units on local market
                    orders.append(Order(product, local_bid, -import_qty))

            # Arb: buy locally, export to foreign
            if local_ask is not None and (export_revenue - local_ask) > self.macaron_arb_min_profit:
                buy_qty = min(self.macaron_base_size, limit - position)
                export_qty = min(buy_qty, self.MACARON_CONVERSION_LIMIT, limit + position)
                if buy_qty > 0:
                    orders.append(Order(product, local_ask, buy_qty))
                if export_qty > 0:
                    conversions -= export_qty  # negative = export

            # Passive market making if no arb
            if not orders:
                mid = get_mid_price(od)
                if not np.isnan(mid):
                    spread = 2
                    passive_bid = int(round(mid - spread))
                    passive_ask = int(round(mid + spread))
                    mm_size = self.macaron_base_size
                    if position < limit:
                        orders.append(Order(product, passive_bid, min(mm_size, limit - position)))
                    if position > -limit:
                        orders.append(Order(product, passive_ask, -min(mm_size, limit + position)))

        return ({product: orders} if orders else {}), conversions

    # -----------------------------------------------------------------------
    # Main run
    # -----------------------------------------------------------------------

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0

        # ---- Load persistent state ----
        if state.traderData:
            try:
                loaded = jsonpickle.decode(state.traderData)
                for key, val_dict in loaded.items():
                    if key in self.persistent_data and isinstance(
                        self.persistent_data[key], defaultdict
                    ):
                        for prod, val in val_dict.items():
                            target = self.persistent_data[key][prod]
                            if isinstance(target, deque) and isinstance(val, list):
                                self.persistent_data[key][prod] = deque(
                                    val, maxlen=self.max_history_len
                                )
                            else:
                                self.persistent_data[key][prod] = val
            except Exception as e:
                print(f"[R4] Failed to load traderData: {e}")

        # ---- Phase 1: Update price histories and indicators ----
        for product, listing in state.listings.items():
            if product not in state.order_depths:
                continue
            mid = get_mid_price(state.order_depths[product])
            if not np.isnan(mid):
                self.persistent_data["price_history"][product].append(mid)
                self._update_indicators(product)
                self.persistent_data["regime"][product] = self._get_regime(product)

        # ---- Phase 2: Trading Logic ----

        # KELP
        kelp_orders = self._trade_kelp(state)
        if kelp_orders:
            result["KELP"] = kelp_orders

        # SQUID_INK
        squid_orders = self._trade_squid(state)
        if squid_orders:
            result["SQUID_INK"] = squid_orders

        # RAINFOREST_RESIN: simple mid-price market making
        if "RAINFOREST_RESIN" in state.order_depths:
            od = state.order_depths["RAINFOREST_RESIN"]
            mid = get_mid_price(od)
            position = get_position("RAINFOREST_RESIN", state)
            limit = self.POSITION_LIMITS["RAINFOREST_RESIN"]
            if not np.isnan(mid):
                fair = round(mid)
                resin_orders = []
                if od.sell_orders:
                    best_ask = min(od.sell_orders.keys())
                    if best_ask < fair:
                        qty = min(abs(od.sell_orders[best_ask]), limit - position)
                        if qty > 0:
                            resin_orders.append(Order("RAINFOREST_RESIN", best_ask, qty))
                if od.buy_orders:
                    best_bid = max(od.buy_orders.keys())
                    if best_bid > fair:
                        qty = min(od.buy_orders[best_bid], limit + position)
                        if qty > 0:
                            resin_orders.append(Order("RAINFOREST_RESIN", best_bid, -qty))
                if resin_orders:
                    result["RAINFOREST_RESIN"] = resin_orders

        # Picnic Baskets
        basket_orders = self._trade_baskets(state)
        result.update(basket_orders)

        # Volcanic Rock / Vouchers
        volcanic_orders = self._trade_volcanic(state)
        result.update(volcanic_orders)

        # Magnificent Macarons
        macaron_orders, macaron_conversions = self._trade_macarons(state)
        result.update(macaron_orders)
        conversions += macaron_conversions

        # ---- Phase 3: Serialize state ----
        serializable = {}
        for key, val_dict in self.persistent_data.items():
            serializable[key] = {}
            if isinstance(val_dict, defaultdict):
                for prod, val in val_dict.items():
                    serializable[key][prod] = list(val) if isinstance(val, deque) else val
        try:
            trader_data = jsonpickle.encode(serializable)
        except Exception as e:
            print(f"[R4] Encoding error: {e}")
            trader_data = ""

        return result, conversions, trader_data
