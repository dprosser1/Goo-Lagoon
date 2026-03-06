from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json

class Trader:
    def __init__(self):
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50}

    def run(self, state: TradingState):
        # Initialize result and conversions
        result = {}
        conversions = 0
        trader_data = {"prices": {}}

        # Load historical data
        if state.traderData:
            trader_data = json.loads(state.traderData)

        # Process each product
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders = []
            
            # Calculate acceptable price based on product
            if product == "RAINFOREST_RESIN":
                acceptable_price = self.calculate_resin_price(order_depth)
            elif product == "KELP":
                acceptable_price = self.calculate_kelp_price(order_depth, trader_data, state.timestamp)
            elif product == "SQUID_INK":
                acceptable_price = self.calculate_squid_price(order_depth, trader_data, state.timestamp)
            else:
                continue  # Skip unknown products

            # Get current position
            current_position = state.position.get(product, 0)
            
            # Generate orders
            if len(order_depth.sell_orders) > 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if best_ask < acceptable_price:
                    buy_qty = min(-best_ask_amount, self.position_limits[product] - current_position)
                    if buy_qty > 0:
                        orders.append(Order(product, best_ask, buy_qty))

            if len(order_depth.buy_orders) > 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if best_bid > acceptable_price:
                    sell_qty = min(best_bid_amount, self.position_limits[product] + current_position)
                    if sell_qty > 0:
                        orders.append(Order(product, best_bid, -sell_qty))

            result[product] = orders

            # Update price history
            mid_price = (best_bid + best_ask) / 2 if order_depth.buy_orders and order_depth.sell_orders else None
            if mid_price:
                if product not in trader_data["prices"]:
                    trader_data["prices"][product] = []
                trader_data["prices"][product].append({
                    "timestamp": state.timestamp,
                    "price": mid_price
                })

        # Truncate historical data to last 100 entries per product
        for product in trader_data["prices"]:
            trader_data["prices"][product] = trader_data["prices"][product][-100:]

        return result, conversions, json.dumps(trader_data)

    def calculate_resin_price(self, order_depth: OrderDepth) -> float:
        # Midprice strategy for stable product
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
        return (best_bid + best_ask) / 2 if best_bid and best_ask else 10

    def calculate_kelp_price(self, order_depth: OrderDepth, trader_data: dict, timestamp: int) -> float:
        # SMA strategy with 5-period window
        price_history = trader_data["prices"].get("KELP", [])
        window = [entry["price"] for entry in price_history[-5:]]
        if len(window) >= 5:
            return sum(window) / 5
        # Fallback to midprice if not enough data
        return self.calculate_resin_price(order_depth)

    def calculate_squid_price(self, order_depth: OrderDepth, trader_data: dict, timestamp: int) -> float:
        # Momentum detection strategy
        price_history = trader_data["prices"].get("SQUID_INK", [])
        if len(price_history) >= 3:
            recent = [entry["price"] for entry in price_history[-3:]]
            if recent[-1] > recent[-2] and recent[-2] > recent[-3]:
                # Upward momentum - bias higher
                return recent[-1] * 1.02
            elif recent[-1] < recent[-2] and recent[-2] < recent[-3]:
                # Downward momentum - bias lower
                return recent[-1] * 0.98
        # Fallback to midprice if not enough data
        return self.calculate_resin_price(order_depth)
