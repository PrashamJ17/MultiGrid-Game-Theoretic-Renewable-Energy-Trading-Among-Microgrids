# Model specification

Status: sections 1–10 are implemented and tested (Phases 1–4).

## 1. Players, time and units

- Players: microgrids `i ∈ {MG1, …, MGN}`; the MVP has N = 4 and up to 6 are supported.
- Time: intervals `t = (day, hour)` with `hour ∈ {0, …, 23}` and Δt = 1 h. The MVP is one day (24 intervals, 96 microgrid-hour observations).
- Energy: every quantity (generation `G`, demand `D`, surplus `S`, deficit `R`, grid import `B`, grid export `X`, trade `q`) is **kWh in that interval**.
- Money: INR. Unit prices and fees: INR/kWh.
- Tolerance: `tol = 1e-8` (configurable, 0 < tol < 1e-2).

## 2. Physical classification (no storage)

```
S[i,t] = max(G[i,t] − D[i,t], 0)
R[i,t] = max(D[i,t] − G[i,t], 0)
role[i,t] = seller if S > tol; buyer if R > tol; balanced otherwise
```

S and R are never both positive, and S − R = G − D holds identically.

## 3. Energy identity

For every player and interval, with every flow ≥ 0:

```
G + B + Σ_j q[j,i] = D + X + Σ_j q[i,j]
```

`energy.check_player_balance` enforces this with tolerance
`tol · max(1, Σ|flows|)`. It also forbids simultaneous grid import and
export, and simultaneous P2P buying and selling.

## 4. Tariffs and fees

`TariffSchedule` holds `b[h]` (retail import price) and `a[h]` (export
remuneration) for each hour of the day. The default is flat: b = 8, a = 3.
If `b[h] < a[h]`, an `EconomicWarning` is raised.

Seller and buyer fees `f_s`, `f_b` are flat per kWh and recorded on every
trade. They are collected by an external operator:

```
seller net receipt per kWh   = p − f_s
buyer delivered price per kWh = p + f_b
unit trade surplus g          = b − a − f_s − f_b
```

## 5. Utility-only baseline

```
B = R,  X = S
C_base[i,t] = b[t]·R[i,t] − a[t]·S[i,t]     (signed: positive = pays)
```

The baseline is the disagreement point of the cooperative bargaining and
will be the outside option in the competitive game. Golden fixture (day 0,
hour 12): C_base = (−60, 120, −30, 200), total 230 INR.

## 6. Shared settlement (`settlement.settle_market`)

Every mechanism produces only a trade log `{(i, j, q_ij, p_ij, f_s, f_b)}`.
Settlement then derives:

```
P2P_in = Σ_j q[j,i]       P2P_out = Σ_j q[i,j]
B = R − P2P_in            X = S − P2P_out
purchase = Σ_j p q[j,i]   sales = Σ_j p q[i,j]
fees = Σ_j f_b q[j,i] + Σ_j f_s q[i,j]
C = b·B − a·X + purchase − sales + fees
U = C_base − C
```

`validate_market` re-derives everything from the trade log and enforces:

| Check | Rule |
|---|---|
| Eligibility | Sellers have S > tol, buyers R > tol, quantities > tol (no phantom trades), at most one trade per pair |
| Capacity | `Σ_j q_ij ≤ S_i + tol` and `Σ_i q_ij ≤ R_j + tol` |
| Energy | the section 3 identity for every player |
| Money | Each component, net cost and gain matches its formula |
| Payment cancellation | `Σ_i purchase_i = Σ_i sales_i` |
| Fee bookkeeping | `Σ_i fees_i = Σ_trades q (f_s + f_b)` |
| Community gain | `Σ_i U_i = Σ_trades q (b − a − f_s − f_b)` |
| Voluntary participation | `p + f_b < b − tol`, `p − f_s ≥ a − tol`, and `U_i ≥ −tol` for every player |

## 7. Cooperative market (`cooperative.solve_cooperative`)

**Stage A: allocation.** Eligible edges are seller-buyer pairs with
`g_ij > tol`. The social LP is:

```
maximise   Σ g_ij q_ij
subject to Σ_j q_ij ≤ S_i,   Σ_i q_ij ≤ R_j,   q_ij ≥ 0
```

It is solved with `scipy.optimize.linprog(method="highs")`. A non-zero
status raises `SolverError`, so no result is ever produced from a failed
solve.

**Tie rule.** Among optimal allocations, the reported one is the
lexicographic maximum of `q` in (seller ID, buyer ID) order over the exact
optimal face. The face is read from the primary LP's duals by complementary
slackness: rows with a nonzero dual are binding in every optimum, and edges
with a positive reduced cost are zero in every optimum. Then `|E|` LPs
maximise one edge each, with earlier edges fixed.

Quantities are rounded to 1e-9 kWh, any float-level capacity excess is
trimmed, and the realised welfare is checked against the optimum. Under
uniform `g` (always the case in the MVP), the result must equal the sorted
greedy rule (seller ID, then buyer ID); otherwise greedy welfare must not
exceed the LP optimum. A violation raises `SolverError`.

**Stage B: bilateral weighted Nash bargaining** on each fixed quantity:

```
r_s = a + f_s,   r_b = b − f_b
ΔU_s = q (p − r_s),   ΔU_b = q (r_b − p)
p* = argmax ΔU_s^α ΔU_b^(1−α) = (1 − α) r_s + α r_b,   0 < α < 1
```

The seller receives the share α of the bilateral surplus. If
`r_b − r_s ≤ tol`, the trade is rejected.

Settlement then checks that the community gain equals the LP welfare, which
guards against fees being counted twice.

**Interpretation:**
- The tie rule does not change total welfare, but it can change individual volumes and gains on the long side of the market. In the golden fixture, MG2 could optimally receive anywhere from 5 to 15 kWh; the ID rule gives it 15.
- This is *bilateral* bargaining on a centrally allocated quantity, not multilateral Nash bargaining or a coalition solution.
- The optimal allocation is an optimisation benchmark, not a Nash equilibrium.
- Changing α only redistributes the fixed total gain `Σ q g`.

**Golden results:**
- Trades MG1→MG2 15, MG1→MG4 5, MG3→MG4 10 kWh, all at INR 5.50.
- C = (−110, 82.5, −55, 162.5), total 80. U = (50, 37.5, 25, 37.5), total 150. Jain fairness 0.94736842.
- With fees 1 + 1: total gain 90, fees 60.
- With f_s + f_b ≥ 5: no trade.

## 8. Competitive market (`competitive.solve_competitive`)

A finite simultaneous-move **seller price game with deterministic buyer
responses** (blueprint section 8).

**Players:** every seller of the interval (S_i > tol). Buyers are not
strategic.

**Strategy sets:** the posted gross prices on the tick `price_step_inr`
that are individually rational:

```
P_i = { k · step :  k · step − f_s ≥ a  and  k · step + f_b < b − tol }
```

The default (a = 3, b = 8, no fees, step 0.5) is {3.0, 3.5, …, 7.5}. At the
lowest price a seller earns exactly its export value. If P_i is empty
(fees absorb the spread), there is no game and no trade.

**Buyer rule:**
1. Delivered price = p_i + f_b.
2. Each buyer sorts acceptable offers by (delivered price, seller ID).
3. Buyers are processed in ascending ID. Each takes as much as possible from its first offer, then the next, never exceeding the remaining surplus or its own deficit.
4. An offer is acceptable only if the delivered price is < b − tol.
5. Unserved deficit is imported; unsold surplus is exported.

**Payoff:** `U_i(p) = Σ_j (p_i − a − f_s) q_ij(p)`, the gain over exporting
the same energy.

**Pure Nash equilibrium:**

```
regret_i(p) = max_{p_i' ∈ P_i} U_i(p_i', p_−i) − U_i(p)
p is a pure NE  ⇔  max_i regret_i(p) ≤ tol
```

The complete payoff table over `P_1 × … × P_m` is computed, with each
profile simulated once (this is the payoff cache). Regrets are then taken
along every axis, so every unilateral deviation of every seller in every
profile is evaluated. `examined_profiles = Π |P_i|` is recorded.

**Selection and reporting:**
- All pure NE are reported (`equilibrium_count`). The settled one has the lowest sum of posted prices, then the lexicographically smallest price vector.
- If there is no pure NE, the status is `NO_PURE_NE` and **no P2P trade is settled**, because the game makes no prediction for that interval. The minimum-maximum-regret profile is recorded as `not_nash_profile`, with its positive regret, and is labelled NOT NASH.
- If there are more sellers than `max_competitive_sellers` (default 3), the status is `ENUMERATION_CAP_EXCEEDED`, a `CombinatorialWarning` is emitted, and nothing is claimed or settled.
- The selected equilibrium's trades are settled through section 6.

**Properties verified by tests:**
- The one-seller game selects the top acceptable price (7.5, utility 45).
- Reported equilibria and all payoffs match an independent reference enumeration.
- A capacity-constrained price war (two 10 kWh sellers, one buyer needing 12 kWh) has no pure NE.
- A Bertrand game (two 20 kWh sellers, 10 kWh demand) has exactly three pure NE, and the selection is (3.0, 3.0).
- With the ID tie rule, two identical sellers get an asymmetric equilibrium set and payoffs.
- The competitive community gain never exceeds the LP optimum.

**Golden fixture:**
- 100 profiles; unique pure NE (MG1 7.50, MG3 7.50), because demand exceeds supply.
- Same 30 kWh as the cooperative market, so the community cost (80) and gain (150) coincide.
- The split differs: 90 / 7.5 / 45 / 7.5 (Jain 0.549 vs 0.947 cooperative).
- These numbers are computed by code; the blueprint gives none.

**Interpretation:**
- This is not a bi-level Stackelberg equilibrium; buyers follow a fixed rule.
- The ID tie rule is a modelling assumption that affects equilibria.
- Discrete prices and the no-NE settlement policy are documented modelling choices.

## 9. Comparison and metrics (`simulation.compare_all`, `metrics.compute_metrics`)

`compare_all(profiles, cfg)` does the following:
1. Validates the profiles once and builds the `HourState`s once.
2. Runs the baseline, cooperative and competitive mechanisms on those identical states, timing each.
3. Checks that the caller's frame is unchanged after every mechanism.
4. Computes each mechanism's metrics with the cooperative LP optimum `Σ_t W*_t` as the efficiency reference.

Metrics (N/A means the value is reported as `None`, never as a misleading number):

```
net community cost  C      = Σ_{i,t} C[i,t]
total gain          U      = C_base − C
cost reduction %           = 100 · U / C_base                  (N/A if C_base ≤ tol)
P2P volume          V      = Σ_trades q
average price              = Σ_trades q·p / V                  (N/A if V ≤ tol)
operator fees              = Σ_trades q·(f_s + f_b)
Jain fairness              = (Σ_i U_i)² / (n Σ_i U_i²)          (N/A if all U_i ≤ tol)
efficiency                 = U / Σ_t W*_t                       (N/A if the optimum ≤ tol)
competitive only: counts of pure-NE / NO_PURE_NE / cap intervals,
                  max regret at settled equilibria, worst NOT NASH regret
```

Every interval also produces a solver-diagnostics record (blueprint
section 5), exported as CSV and JSON.

## 10. Data contracts

See the README section "Data contracts" and the column tuples in
`src/microgrid/models.py` (`TRADE_CSV_COLUMNS`, `SETTLEMENT_CSV_COLUMNS`).
