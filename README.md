# Goo-Lagoon — IMC Prosperity 3 (2025)

Team: **Goo-Lagoon**
Competition: IMC Prosperity 3 — April 7–22, 2025
Format: 5 rounds × 72 hours, algorithmic + manual trading in a simulated island economy.

---

## Repository Structure

| File | Description |
|------|-------------|
| `datamodel.py` | Official IMC datamodel (Order, TradingState, ConversionObservation, etc.) |
| `Round1Trader.py` | Round 1 refined trader — rolling average market-maker for KELP & RAINFOREST_RESIN |
| `Trader.py` | Round 1 baseline/starter trader (simpler SMA + momentum version) |
| `round2trader.py` | Round 2 trader — adaptive regime detection, Bollinger Bands, Picnic Basket arb |
| `r3trader.py` | Round 3 trader — extends Round 2 with VOLCANIC_ROCK and options-based voucher trading |
| `round4trader.py` | Round 4 trader — adds MAGNIFICENT_MACARONS cross-market arbitrage with sunlight regime |
| `round5trader.py` | Round 5 trader — adds Olivia insider copy-trading; simplified macaron strategy |
| `round-1-island-data-bottle.zip` | Official Round 1 market data provided by IMC |

---

## Competition Overview

IMC Prosperity 3 ran for 15 days across 5 rounds, with 12,000+ teams worldwide. Each round
lasted 72 hours and introduced new tradeable products, building from 3 products in Round 1 to
15 by the final round. The scoring currency was SeaShells. Teams submitted a Python algorithm
each round that would be run against IMC's bots in continuous double-auction simulation.

Products by round:

- **Round 1:** RAINFOREST_RESIN, KELP, SQUID_INK
- **Round 2:** + CROISSANT, JAM, DJEMBE, PICNIC_BASKET1, PICNIC_BASKET2
- **Round 3:** + VOLCANIC_ROCK, VOLCANIC_ROCK_VOUCHER_9500/9750/10000/10250/10500
- **Round 4:** + MAGNIFICENT_MACARONS
- **Round 5:** All 15 products + de-anonymized market counterparties

---

## Round 1

### Products
RAINFOREST_RESIN · KELP · SQUID_INK

### Strategy

**RAINFOREST_RESIN** behaves as a stable, mean-reverting asset centered near 10,000 SeaShells.
We applied a simple mid-price market-making approach: calculate the fair value as the midpoint
of the best bid and best ask, then buy anything below it and sell anything above it.

**KELP** showed moderate volatility with no persistent trend. We used a 5-period simple
moving average (SMA) as the fair value estimate and traded aggressively when market prices
diverged from it. A risk-aversion factor reduced order sizes proportionally as the position
approached its limit (±20 units).

**SQUID_INK** displayed the highest volatility of the three products, with sharp random
spikes and no exploitable long-term trend. The strategy applied a 3-period momentum detector:
if prices rose consecutively, we biased our fair-value estimate upward by 2% (expecting
continuation); if prices fell consecutively, we biased downward by 2%. This proved unstable
over longer horizons.

### Discussion

The two Round 1 files (`Trader.py` and `Round1Trader.py`) reflect an iterative improvement
process during the round. `Trader.py` was the initial submission; `Round1Trader.py` was the
refined version that introduced the rolling average framework, configurable spread thresholds,
and a unified PnL tracking system. The core challenge was the absence of historical data at
the start of each day — strategies relying on lookback windows are effectively blind for the
first N timesteps.

SQUID_INK's momentum strategy was too reactive to noise. The 2% price bias created large
unrealized positions that unwound at a loss when spikes reversed. The product likely requires
mean-reversion after spikes rather than momentum following.

### Reflection

The market-making approach on RAINFOREST_RESIN was the most reliable earnings source in
Round 1. The hardcoded fair value of ~10,000 proved robust because the product genuinely
mean-reverts to that level. KELP's SMA approach worked but was slow to adapt on new trading
days. The biggest lesson: SQUID_INK needed to be treated as a volatility product, not a
momentum one.

### Improvement Upon Strategy

- Replace SQUID_INK momentum with a volatility spike mean-reversion: buy when the price
  falls more than N standard deviations from a rolling mean, sell when it recovers.
- Implement a warm-up period for SMA-based strategies so no orders are placed until at least
  one full window of data has been collected.
- Use a tighter risk-aversion curve on KELP to avoid large directional exposure.

---

## Round 2

### Products
All Round 1 products + CROISSANT · JAM · DJEMBE · PICNIC_BASKET1 · PICNIC_BASKET2

### Strategy

The Round 2 trader introduced an **adaptive regime detection** framework. Before placing any
order, the algorithm computed:

1. **Current volatility** — standard deviation of recent mid-price changes over a 20-period
   window.
2. **Long-term average volatility** — an exponential moving average (EMA) of the volatility
   series used as a baseline.
3. **Trend slope** — the rate of change of a 50-period moving average, compared to its value
   5 periods earlier.
4. **Regime classification** — "Trending Up", "Trending Down", "Ranging High Vol", or
   "Ranging Low Vol".

**KELP** used Bollinger Band mean-reversion, active only in "Ranging" regimes. Band width was
adapted by the volatility ratio (current vol / average vol), widening bands during high-vol
periods to avoid whipsaw.

**SQUID_INK** switched to a momentum strategy active only in "Trending" regimes, buying into
upward trends and selling into downward trends when the price move exceeded 1.5× the current
volatility.

**Picnic Baskets (PB1 and PB2)** were treated as synthetic ETFs. PB1 = 6 Croissants + 3 Jams
+ 1 Djembe; PB2 = 4 Croissants + 2 Jams. The strategy computed the "fair basket value" from
the mid-prices of components and executed index arbitrage when the basket price diverged by
more than 20 SeaShells from fair value. Implementation bugs in the component position limit
management limited profitability.

### Discussion

The regime-detection framework was a meaningful architectural upgrade over Round 1's static
logic. By conditioning strategies on market state, the algorithm reduced false signals during
choppy periods (preventing Bollinger breakout trades during trending markets) and avoided
trend-following during mean-reverting conditions.

The Picnic Basket arbitrage was theoretically sound but had two practical issues. First,
the component position limits were not correctly factored into the basket sizing — we could
attempt to buy a basket larger than our available component capacity would allow. Second,
the 20 SeaShell spread threshold was calibrated too conservatively, causing many genuine
arbitrage windows to be missed.

### Reflection

The adaptive regime system was the most important structural contribution of Round 2. Getting
a cleaner separation between "this product is trending" and "this product is ranging" was
essential for the strategies to stop interfering with each other. The basket arbitrage captured
genuine edge in theory but execution quality hurt realized PnL. Additionally, the use of
`jsonpickle` for state serialization introduced occasional decoding failures that would reset
all indicators mid-competition.

### Improvement Upon Strategy

- Fix the component capacity check in basket arbitrage to ensure hedge ratios respect each
  component's individual position limit simultaneously.
- Lower the basket spread threshold to 10–12 SeaShells after backtesting to capture more
  arb events.
- Add a fallback JSON serialization path that uses simple `json.dumps()` when `jsonpickle`
  fails, preserving at minimum the price history.
- For CROISSANT and JAM traded individually, implement a light market-making overlay since
  they have wide bid-ask spreads.

---

## Round 3

### Products
All Round 2 products + VOLCANIC_ROCK · VOLCANIC_ROCK_VOUCHER_9500/9750/10000/10250/10500

### Strategy

**VOLCANIC_ROCK_VOUCHER** contracts were modeled as **European call options** using the
Black-Scholes formula with the underlying being VOLCANIC_ROCK and a fixed expiry of
approximately 5 days (competition convention). The approach:

1. For each voucher, observe its market mid-price and back-solve for **implied volatility (IV)**
   using Newton-Raphson iteration.
2. Compute the **average IV** across all five voucher strikes as a smoothed volatility estimate.
3. Re-price each voucher using this average IV to obtain a model fair value.
4. Buy vouchers where the market ask is more than 3% below fair value.
5. Sell vouchers where the market bid is more than 3% above fair value.

**VOLCANIC_ROCK** itself was traded based on directional signals derived from the IV surface:
a sudden rise in average IV indicated uncertainty and was used to reduce net exposure.

All prior Round 2 strategies were maintained unchanged.

### Discussion

The Black-Scholes model was a natural fit for the voucher structure. The key decision was
whether to track each voucher's IV independently or use an aggregated IV. Using an average
across strikes worked well because it smoothed out strikes where the market was thin and
spread costs made individual IV estimates unreliable.

The most notable event of Round 3 was the accidental accumulation of a large short position
in VOLCANIC_ROCK through the baseline placeholder logic. This short happened to align with
a market downturn and generated significant profit — a result of luck rather than design.
This highlighted how important VOLCANIC_ROCK directional exposure was to overall round
performance.

### Reflection

The options framework was the right model for the vouchers, and the result demonstrated
meaningful alpha from fair-value pricing relative to uninformed bots. However, the
VOLCANIC_ROCK spot strategy was essentially a placeholder (naive threshold-based buy/sell at
10,000) and did not exploit the IV signal coherently. The lucky short was not reproducible.
The team should have designed a delta-hedging overlay: tracking net delta across all held
voucher positions and neutralizing it via offsetting VOLCANIC_ROCK trades.

### Improvement Upon Strategy

- Implement a proper **delta hedge**: compute the Black-Scholes delta for each held voucher,
  sum net delta across the book, and trade VOLCANIC_ROCK to bring net delta toward zero.
- Track the **IV skew** (IV by strike) to detect when the market is pricing tail risk into
  specific strikes — overpriced OTM puts/calls are a separate arb opportunity.
- Set a minimum liquidity filter: only trade vouchers where the market volume supports at
  least a 3-unit fill without excessive slippage.
- Remove the hardcoded 10,000 threshold from VOLCANIC_ROCK spot; replace with a
  dynamic fair-value derived from the options surface.

---

## Round 4

### Products
All Round 3 products + MAGNIFICENT_MACARONS

### Strategy

**MAGNIFICENT_MACARONS** could be traded either on the local island exchange (via the standard
order book) or cross-market with a foreign island via a **conversion mechanism**. The
`ConversionObservation` provided:

- `bidPrice` / `askPrice`: foreign island's live market quotes
- `transportFees`: cost per unit shipped
- `exportTariff` / `importTariff`: per-unit taxes on cross-border trades
- `sunlightIndex`: a continuous measure of recent sunlight hours (affects production cost)
- `sugarPrice`: cost of sugar inputs (also affects cost)

Effective costs:

- **Import cost** = `askPrice + importTariff + transportFees`
- **Export revenue** = `bidPrice - exportTariff - transportFees`

Two regimes were implemented:

1. **Normal sunlight (sunlightIndex ≥ 2500):** Two-way arbitrage. Import and immediately
   sell locally when `localBid − importCost > 1`. Buy locally and export when
   `exportRevenue − localAsk > 1`. Passive market-making when neither arb was available.

2. **Low sunlight (sunlightIndex < 2500):** Accumulation mode. Build a long position locally
   and via imports, anticipating that reduced production would push local prices higher.
   Avoid exporting inventory.

The position limit was 75 units; conversions were capped at 10 per timestep.

### Discussion

The conversion mechanism was the most complex new mechanic in Prosperity 3. A key insight
shared by top teams was the existence of a hidden taker bot that aggressively filled locally-
offered orders priced attractively relative to an internal fair value. This meant that placing
a passive ask slightly above import cost would be hit by the bot for an expected gain of ~3
SeaShells per unit — not captured in naive backtesting because the backtester did not simulate
the taker bot.

Our sunlight-based accumulation regime was conceptually sound but difficult to calibrate
without a reliable backtesting environment that included conversion mechanics. The strategy
was incomplete at submission time.

### Reflection

Round 4 highlighted a testing infrastructure gap. The publicly available backtesting tools
(`prosperity3bt`) did not support the conversion mechanic, which made it impossible to properly
validate the arb logic before submission. The sunlight accumulation model introduced a
directional bet that was harder to size correctly than a pure arbitrage.

The two-way arbitrage was the safer and more reliable component. The accumulation bet depended
on correctly predicting the direction of local prices from an external signal, which introduced
model risk.

### Improvement Upon Strategy

- Use the taker bot insight: post passive limit orders at a fixed offset above import cost
  rather than aggressively crossing the spread. The taker bot provides ~60% fill rate at a
  3 SeaShell premium.
- Decouple the sunlight accumulation strategy from the core arbitrage so each can be toggled
  and sized independently.
- Extend the backtester to simulate conversion requests manually, enabling proper round-trip
  P&L calculation.
- Track `sugarPrice` as a secondary signal alongside `sunlightIndex` to produce a combined
  input cost estimate for better fair-value calculation.

---

## Round 5

### Products
All 15 Round 4 products + de-anonymized counterparty IDs on all market trades.

### Strategy

When counterparty identities became visible in `market_trades`, the strategy introduced
**insider detection and copy-trading**.

**Step 1 — Insider Detection:** For each counterparty and each product, compute a rolling
"good trade" ratio: the fraction of their recent trades where buying preceded a price increase,
or selling preceded a price decrease. A trader named **Olivia** showed a statistically
significant good-trade rate well above chance across SQUID_INK and CROISSANT.

**Step 2 — Copy Trading:** When Olivia's good-trade rate exceeded a 60% threshold over the
most recent 30-trade window, her most recent directional action was used as a regime signal:

- If Olivia bought → we take a long position.
- If Olivia sold → we take a short position.

Order size was set to 50% of remaining position capacity, prioritizing the Olivia signal over
the standard SQUID_INK momentum strategy.

**MAGNIFICENT_MACARONS** strategy was simplified to pure two-way arbitrage only, removing the
sunlight accumulation model which had proven overfit and hard to calibrate.

All other Round 4 strategies (KELP Bollinger, baskets, vouchers) were maintained.

### Discussion

The insider identification approach was data-driven and principled. Rather than manually
inspecting trade data, the algorithm automatically scores all observed counterparties and
acts only when the evidence is statistically meaningful. The 60% threshold and 30-trade
window are tunable parameters — a lower threshold increases coverage but also false positives.

The copy-trading on CROISSANT was particularly effective because CROISSANT feeds directly
into the Picnic Basket composition, meaning Olivia's CROISSANT trades likely reflected
information about basket mispricing that the algorithm could benefit from twice (once via
copy-trade, once via basket arb).

A limitation of the approach was latency: since we update Olivia's signal based on trades
observed in the current timestep, we can only act one timestep after her trade. In a fast
market with thin liquidity, this lag reduces capture rate.

### Reflection

Round 5 demonstrated that extracting information from the trading environment itself — not
just from price series — is a powerful source of alpha. The competition was structured so that
a bot with an information advantage could be detected and exploited algorithmically, rewarding
teams who thought beyond price-based indicators.

The late identification of this opportunity (Round 5 of 5) meant that earlier rounds' P&L
could not benefit from the insight. Had counterparty tracking been implemented as an observer
from Round 1 (storing all trade counterparties against outcomes from the start), the team
would have been able to identify Olivia much earlier and act sooner.

The macaron simplification was correct — the sunlight model added complexity without proven
edge. Pure arbitrage is more robust and easier to size.

### Improvement Upon Strategy

- Implement counterparty tracking from Round 1 as a passive observer, so that if IDs are
  ever revealed, a full historical record is available to compute accurate good-trade rates
  immediately.
- Add a Bayesian update mechanism for the good-trade rate: weight recent trades more heavily
  than older ones, so the signal adapts quickly to Olivia's behavior in later rounds.
- Test copy size sensitivity: 50% capacity copy may be too large if Olivia's signal has
  meaningful false-positive rate at the 60% threshold. A smaller fraction (25–30%) may
  achieve a better Sharpe.
- Cross-reference Olivia's CROISSANT trades with Picnic Basket pricing to identify whether
  her edge is informational (about component fair value) or mechanical (exploiting basket
  mispricing that the arb bot has not yet acted on).

---

## Overall Takeaways

The five rounds traced a clear progression from basic market-making to multi-asset statistical
arbitrage to options pricing to cross-market arbitrage to information-based trading. Key
lessons:

1. **Product characterization matters first.** Identifying whether a product is mean-reverting,
   trending, or driven by external signals should precede any strategy design.

2. **Infrastructure reliability is as important as strategy quality.** State serialization
   failures, backtesting limitations, and position limit bugs cost real SeaShells regardless
   of how sophisticated the underlying strategy was.

3. **Simple arbitrage beats complex predictions.** The most reliable edge in every round came
   from mechanical arbitrage (basket arb, conversion arb) rather than directional forecasting.

4. **Market microstructure is exploitable.** The hidden taker bot in Round 4 and Olivia in
   Round 5 both rewarded teams who looked beyond raw prices at the structure of how trades
   were executing.

---

*IMC Prosperity 3 · Team Goo-Lagoon · 2025*
