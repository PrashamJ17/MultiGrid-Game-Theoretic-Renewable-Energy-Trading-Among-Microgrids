# Eight-minute demo script

This follows the blueprint section 16 plan. Before you start:

1. Open two terminals in the repository root.
2. Run `python -m streamlit run app.py` and open the browser tab it prints.
3. Keep the second terminal for step 6.

All values below were checked in the dashboard on the tested machine.

## 0:00–1:00 · Motivation (Overview tab, golden fixture)

- **Show:** the sidebar scenario is "Golden one-hour fixture". Point at the yellow banner: synthetic data, illustrative tariffs, not a trading system.
- **Say:** four solar microgrids, one midday hour. Exporting surplus earns only INR 3/kWh, but importing costs INR 8/kWh. That **INR 5/kWh spread** is what local trading can capture.
- **Point at:** the metric tiles. Tariff spread 5.00; fees 0.00; Baseline net cost **230.00**.

## 1:00–2:00 · Profiles and classification (Profiles tab)

- **Show:** the net-position bars. MG1 +20 and MG3 +10 are sellers (surplus); MG2 −15 and MG4 −25 are buyers (deficit).
- **Say:** no storage, so in any hour a microgrid is a seller or a buyer, never both. Total surplus is 30 kWh against a total deficit of 40 kWh.
- **Optional:** switch the scenario to "Synthetic 24 h, seed 42" to show the daily curves. Surplus exists only around 07:00–16:00. Switch back to golden afterwards.

## 2:00–3:15 · No P2P vs cooperative (Overview → Transactions → Comparison)

- **Overview:** Cooperative net cost **80.00**, i.e. −150.00 vs baseline.
- **Transactions (Cooperative):** three trades. MG1→MG2 15 kWh, MG1→MG4 5 kWh and MG3→MG4 10 kWh, all at **INR 5.50**.
- **Say:**
  - The LP fixes the quantities (30 kWh, every kWh of surplus).
  - Each pair then splits its INR 5/kWh by Nash bargaining around the grid prices, which lands on the midpoint, 5.50.
  - The **INR 150** community benefit is (8 − 3) × 30.
- **Comparison:** gains of 50 / 37.5 / 25 / 37.5; Jain fairness 0.947.

## 3:15–4:45 · Seller strategies and Nash verification (Game analysis tab)

- **Golden hour:**
  - Show the green box: "Verified pure Nash equilibrium: MG1=7.50, MG3=7.50 (1 pure NE among 100 profiles)".
  - Show the regret heatmap: one NE cell in the top-right.
- **Say:**
  - Sellers choose posted prices from 3.00 to 7.50. Buyers take the cheapest offer first.
  - Every one of the 10 × 10 price pairs is checked against every unilateral deviation.
  - Demand exceeds supply here, so both sellers sell everything at any price and the only equilibrium is the top price. The total gain is the same 150, but sellers capture 135 of it (Jain 0.549).
- **Switch to the 24 h seed-42 scenario** and pick **day 0, hour 12** under "Interval to analyse":
  - The red banner reads **NO PURE EQUILIBRIUM FOUND**, and the diagnostic MG1=7.00, MG3=6.00 is labelled **NOT NASH**.
  - In the utility curves, MG1 would earn more at 7.50.
  - **Say:** with oversupply, capacity-limited sellers undercut each other in a cycle. We report that honestly, and that hour is not settled.

## 4:45–5:45 · Fees and the no-benefit threshold (sidebar, golden scenario)

1. Set **Seller fee 1.00** and **Buyer fee 1.00**.
   - Cooperative net cost becomes **140.00** (gain 90, with INR 60 going to the operator). Transactions still show 5.50, with seller net 4.50 and buyer delivered 6.50.
2. Set both fees to **2.50**.
   - All three mechanisms show **230.00**.
   - Transactions shows "No P2P trades".
   - **Say:** once the fees reach the spread, no trade is beneficial.
3. Reset the fees to 0.

## 5:45–6:45 · Invariant checks and tests (second terminal)

```bash
python -m pytest -q
python scripts/audit_exports.py results/experiments/E1
```

**Expected:**
- `345 passed`
- `22 passed, 0 failed`

**Say:**
- Each settlement checks energy balance, that P2P payments cancel, fees, capacities and voluntary participation.
- The audit script re-derives all of that, plus every equilibrium label, from the exported CSVs without importing the package.

## 6:45–8:00 · Findings, limitations, extensions

- **Show:** `docs/figures/fig07_fee_sensitivity.png` and `fig12_seeds.png`, or the Comparison tab on the 24 h day.
- **Findings:**
  - Cooperative gain is 1,259.93 INR on the reference day and stable across seeds (1,284.80 ± 30.72).
  - Competitive gain is 862.08 INR, more variable (898.80 ± 124.02), and seller-heavy.
  - The entire gap comes from 1–3 oversupplied hours a day without a pure equilibrium.
  - Moderate fees can remove the price war (Fig. 7).
- **Limitations:**
  - synthetic data
  - no storage or network
  - buyer tie rule by ID
  - discrete prices
  - exact enumeration capped at three sellers
- **Extensions:** storage, network constraints, multilateral bargaining or Shapley allocation, learning agents.

## Viva questions (blueprint section 16) with short answers

- **Players, actions, payoffs?**
  - *Cooperative:* there is no strategic game. An LP allocates quantities, then each pair bargains a price.
  - *Competitive:* the sellers are the players and their actions are posted prices on the tick. The payoff is `Σ (p − a − f_s)·q`, which is gain over exporting.
- **Why is the optimal allocation not a Nash equilibrium?** It solves a central planner's problem. Nobody's unilateral deviation is checked, because there are no strategies.
- **What is bargained over?** Only the price of each fixed bilateral quantity. The quantities come from the LP.
- **How do tariffs set the disagreement payoffs?** If the pair doesn't trade, the seller exports at *a* (minus nothing) and the buyer imports at *b*. That gives reservation prices `a + f_s` and `b − f_b`.
- **Why might there be no pure equilibrium?** With oversupply, the lowest-priced seller fills the demand. The other seller wants to undercut, but the undercut seller then prefers to jump back to the top price for the residual demand, so best responses cycle.
- **How does the tie rule matter?** Buyers break equal prices by seller ID, so identical sellers can get different equilibrium payoffs. A test demonstrates this.
- **Why do P2P payments cancel?** Every rupee a buyer pays, a seller receives. Only the grid bill and the fees leave the community.
- **What grid physics is omitted?** Line flows, voltage, losses, storage, forecasting and dispatchable generation.
