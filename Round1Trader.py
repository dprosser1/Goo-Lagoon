import json
from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order
import statistics

class Trader:
    def __init__(self):
        # --------------------- Configuration ----------------------
        # Position & Spread Limits
        self.position_limits = {"RAINFOREST_RESIN": 20, "KELP": 20, "SQUID_INK": 20}
        self.spread_thresholds = {"RAINFOREST_RESIN": 2, "KELP": 3, "SQUID_INK": 4}
        self.history_length = 150  # Store up to 100 mid-prices

        # Rolling average window (ticks) - Reduced for faster response
        self.rolling_window = 5  # Changed from 10 to 5 for quicker adaptation to price changes

        # Target spreads for buy/sell decisions (fixed spread around rolling average)
        self.target_spreads = {"RAINFOREST_RESIN": 1, "KELP": 1, "SQUID_INK": 1}

        # Risk parameter
        self.risk_aversion_factor = 0.5

        # Define tradable products
        self.tradable_products = ["KELP", "RAINFOREST_RESIN"]

    # -------------------------------------------------------------------------
    # ----------------------- Rolling Average Utilities -----------------------
    # -------------------------------------------------------------------------

    def calculate_rolling_average(self, price_history: List[Dict], window: int) -> float | None:
        """
        Calculate the simple moving average of mid-prices over the specified window.
        Uses all available data if fewer prices than window size are present.
        """
        if not price_history:
            return None
        windowed_prices = price_history[-window:]
        prices = [p["price"] for p in windowed_prices]
        return statistics.mean(prices)

    def calculate_price_targets(self, product: str, price_history: List[Dict]) -> Tuple[float | None, float | None]:
        """
        Calculate buy and sell targets based on the rolling average.
        Buy target = rolling average - spread, Sell target = rolling average + spread.
        """
        rolling_avg = self.calculate_rolling_average(price_history, self.rolling_window)
        if rolling_avg is None:
            return None, None
        spread = self.target_spreads[product]
        buy_target = rolling_avg - spread
        sell_target = rolling_avg + spread
        return buy_target, sell_target

    # -------------------------------------------------------------------------
    # --------------------------- Helper Functions ----------------------------
    # -------------------------------------------------------------------------

    def get_best_bid_ask(self, order_depth: OrderDepth) -> Tuple[float | None, float | None]:
        """Get the best bid and ask prices from the order depth."""
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def update_pnl(self, trader_data: Dict, product: str, state: TradingState):
        """Update realized and unrealized PnL based on trades and current position."""
        pnl_data = trader_data.setdefault("pnl", {})
        position_data = trader_data.setdefault("positions", {})

        realized_pnl = pnl_data.get(product, {}).get("realized", 0.0)
        avg_cost = position_data.get(product, {}).get("avg_cost", 0.0)
        current_pos = state.position.get(product, 0)
        prev_pos_info = position_data.get(product, {"quantity": 0, "avg_cost": 0.0})
        prev_pos_qty = prev_pos_info.get("quantity", 0)

        # Process trades from the last tick
        for trade in state.own_trades.get(product, []):
            qty = trade.quantity
            px = trade.price
            if qty > 0:  # Bought
                total_cost = (prev_pos_qty * avg_cost) + (qty * px)
                new_pos_qty = prev_pos_qty + qty
                avg_cost = total_cost / new_pos_qty if new_pos_qty != 0 else 0
                prev_pos_qty = new_pos_qty
                print(f"Bought {qty} {product} @ {px:.2f}, new avg cost: {avg_cost:.2f}")
            elif qty < 0:  # Sold
                sell_qty = abs(qty)
                profit = (px - avg_cost) * sell_qty
                realized_pnl += profit
                prev_pos_qty += qty
                print(f"Sold {sell_qty} {product} @ {px:.2f}, realized PnL: {profit:.2f}, total Realized: {realized_pnl:.2f}")

        position_data[product] = {"quantity": current_pos, "avg_cost": avg_cost}
        pnl_data.setdefault(product, {})["realized"] = realized_pnl

        # Unrealized PnL
        unrealized_pnl = 0.0
        if current_pos != 0 and product in state.order_depths:
            best_bid, best_ask = self.get_best_bid_ask(state.order_depths[product])
            if best_bid is not None and best_ask is not None:
                mid_p = (best_bid + best_ask) / 2
                unrealized_pnl = (mid_p - avg_cost) * current_pos

        pnl_data[product]["unrealized"] = unrealized_pnl
        pnl_data[product]["timestamp"] = state.timestamp

    # -------------------------------------------------------------------------
    # ------------------- Main Execution / Run Method -------------------------
    # -------------------------------------------------------------------------

    def run(self, state: TradingState):
        result = {}
        conversions = 0  # Not used in this example

        # Load persistent data
        trader_data = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except json.JSONDecodeError:
                print("Error decoding traderData, starting fresh.")
                trader_data = {}

        trader_data.setdefault("prices", {})
        trader_data.setdefault("pnl", {})
        trader_data.setdefault("positions", {})
        trader_data.setdefault("day", state.timestamp // 1000000)

        # Check if new day
        current_day = state.timestamp // 1000000
        if current_day > trader_data.get("day", 0):
            print(f"New day detected: {current_day}")
            trader_data["day"] = current_day

        # Process each product
        for product, order_depth in state.order_depths.items():
            if product not in self.position_limits:
                print(f"Warning: Product {product} not in position limits.")
                continue

            orders: List[Order] = []
            current_position = state.position.get(product, 0)
            position_limit = self.position_limits[product]

            best_bid, best_ask = self.get_best_bid_ask(order_depth)
            spread = float("inf")
            mid_price = None
            if best_bid is not None and best_ask is not None:
                spread = best_ask - best_bid
                mid_price = (best_bid + best_ask) / 2

                # Record mid_price in our price history
                ph = trader_data["prices"].setdefault(product, [])
                ph.append({"timestamp": state.timestamp, "price": mid_price})
                trader_data["prices"][product] = ph[-self.history_length:]

            # Update PnL from any of our trades last tick
            self.update_pnl(trader_data, product, state)

            # Skip trading logic for non-tradable products
            if product not in self.tradable_products:
                result[product] = []
                continue

            # ------------- Get Buy/Sell Targets via Rolling Average -------------
            price_history = trader_data["prices"].get(product, [])
            if not price_history:
                result[product] = []
                continue
            buy_target, sell_target = self.calculate_price_targets(product, price_history)
            if buy_target is None or sell_target is None:
                result[product] = []
                continue

            # Optionally skip trading if spread > threshold
            prod_spread_threshold = self.spread_thresholds.get(product, float("inf"))
            allow_trade = True
            if spread > prod_spread_threshold:
                # Uncomment to skip trading on large spreads
                # allow_trade = False
                pass

            # ------------------- Placing Buy/Sell Orders -------------------
            if allow_trade and (best_bid is not None) and (best_ask is not None):
                # BUY if best_ask <= buy_target
                if best_ask <= buy_target:
                    available_to_buy = position_limit - current_position
                    # Reduce quantity if we already hold a big position
                    risk_factor = (1 - self.risk_aversion_factor *
                                   (current_position / position_limit if current_position > 0 else 0))
                    max_buy_qty = max(0, int(available_to_buy * risk_factor))

                    # How many are available on the ask side
                    potential_buy_volume = 0
                    sorted_asks = sorted(order_depth.sell_orders.items())
                    for px, vol in sorted_asks:
                        if px <= buy_target:
                            potential_buy_volume += abs(vol)
                        else:
                            break
                    buy_qty = min(potential_buy_volume, max_buy_qty)
                    if buy_qty > 0:
                        print(f"BUYING {buy_qty} {product} at {best_ask} (target={buy_target:.2f})")
                        orders.append(Order(product, best_ask, buy_qty))

                # SELL if best_bid >= sell_target
                if best_bid >= sell_target:
                    available_to_sell = current_position + position_limit
                    # Reduce quantity if we have a large negative position
                    risk_factor = (1 - self.risk_aversion_factor *
                                   (-current_position / position_limit if current_position < 0 else 0))
                    max_sell_qty = max(0, int(available_to_sell * risk_factor))

                    potential_sell_volume = 0
                    sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)
                    for px, vol in sorted_bids:
                        if px >= sell_target:
                            potential_sell_volume += abs(vol)
                        else:
                            break
                    sell_qty = min(potential_sell_volume, max_sell_qty)
                    if sell_qty > 0:
                        print(f"SELLING {sell_qty} {product} at {best_bid} (target={sell_target:.2f})")
                        orders.append(Order(product, best_bid, -sell_qty))

            result[product] = orders

        # Serialize trader_data for next tick
        trader_data_str = json.dumps(trader_data)
        return result, conversions, trader_data_str