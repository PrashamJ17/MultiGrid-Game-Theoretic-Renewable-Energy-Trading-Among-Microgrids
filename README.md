# MultiGrid: Game-Theoretic Analysis of Renewable Energy Trading Among Microgrids

An offline, reproducible Python simulator of peer-to-peer (P2P) renewable energy trading among
grid-connected microgrids. It compares:

1. a **utility-only baseline** (no P2P trading),
2. a **cooperative market**: a welfare-maximising LP allocation with bilateral weighted Nash bargaining over
   price, and
3. a **competitive market**: a finite seller-price game with exhaustive pure-Nash-equilibrium verification.

All three run on the **same frozen energy profiles** and go through one shared settlement. The project also
includes a Streamlit dashboard, seven controlled experiments, report figures, report material, a demo script
and an independent audit.

> **Simulation only.** All generation and demand profiles are synthetic. The tariffs (retail INR 8/kWh,
> export INR 3/kWh) are illustrative assumptions, not the tariffs of any real utility. This is a research and
> teaching demonstration, not an electricity trading or grid-control system.

**Team:** Krishna Tyagi, Prasham Jain, Hrishi Bhalaria and Lakshya Srivastava.

**Full report:** [docs/MultiGrid_Project_Report.pdf](docs/MultiGrid_Project_Report.pdf) (editable version:
[.docx](docs/MultiGrid_Project_Report.docx)). **Presentation:** [docs/MultiGrid_presentation.pptx](docs/MultiGrid_presentation.pptx).

## Implementation status

All five phases are complete. **345 tests pass**, and the independent export audit passes **22/22** checks.

| Phase | Scope |
|---|---|
| 1. Foundation | Data contracts and validation, seeded 24 h generator, energy model, utility-only baseline |
| 2. Cooperative market | Shared settlement and validation, HiGHS LP with a duality-based tie rule, bilateral Nash bargaining |
| 3. Competitive market | Seller price game, deterministic buyer rule, exhaustive payoff table, all pure NE, `NO_PURE_NE` and cap reporting |
| 4. Integration | `compare_all` controller, full metrics (explicit N/A), diagnostics export, `compare` CLI, eight-tab dashboard |
| 5. Experiments and handover | E1–E7 runner, 12 report figures, experiment protocol, report material, demo script, independent audit |

## Documents

| Document | Contents |
|---|---|
| [docs/MultiGrid_Project_Report.pdf](docs/MultiGrid_Project_Report.pdf) | The complete project report (59 pages): problem, background, model, cooperative and competitive games with worked payoff matrices, software design and dashboard, verification, experiments E1–E7 with all figures and tables, discussion, conclusions, references and appendices. Editable version: [.docx](docs/MultiGrid_Project_Report.docx) |
| [docs/MultiGrid_presentation.pptx](docs/MultiGrid_presentation.pptx) | 17-slide presentation of the project and its results, with speaker notes |
| [docs/report.md](docs/report.md) | Short report summary: problem, model, verification and the results of E1–E7 with figures |
| [docs/experiment_protocol.md](docs/experiment_protocol.md) | Design rules, the experiment matrix, output files, interpretation rules |
| [docs/model_specification.md](docs/model_specification.md) | Every player, price, payoff, equilibrium and metric definition |
| [docs/demo_script.md](docs/demo_script.md) | 8-minute dashboard demo with expected values, plus viva answers |
| [docs/audit_report.md](docs/audit_report.md) | PASS/FAIL audit against every MVP requirement, with file/line evidence |
| [docs/figures/](docs/figures/) | The 12 report figures (PNG), regenerated from saved results |

## Setup

Requires Python 3.10 or newer. Run every command from the repository root.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Windows PowerShell**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs the dependencies and the `microgrid` package in editable mode.

### Troubleshooting: `No module named 'microgrid'` on macOS

The tests and the dashboard are unaffected by this, because both add `src/` themselves.

- **The cause:** iCloud Drive syncs the Desktop and Documents folders and marks every file inside `.venv` as hidden. Python 3.13 ignores hidden `.pth` files, which breaks the editable install, so `python -m microgrid.cli` fails.
- **Fix 1:** create the venv outside iCloud, for example `python3 -m venv ~/.venvs/multigrid`.
- **Fix 2:** prefix CLI commands with `PYTHONPATH=src`.

## Commands

```bash
# Tests
python -m pytest -q

# Interactive dashboard
python -m streamlit run app.py

# Compare all three mechanisms on one scenario (writes every table with --out-dir)
python -m microgrid.cli compare --config configs/default.json --profiles data/sample_24h.csv --out-dir results/

# Experiments E1-E7, report figures, independent export audit
python -m microgrid.cli experiment all --out-dir results/experiments
python scripts/make_figures.py --results results/experiments --out docs/figures
python scripts/audit_exports.py results/experiments/E1

# Single mechanisms and data generation
python -m microgrid.cli competitive --config configs/default.json --profiles data/sample_24h.csv --out-dir results/
python -m microgrid.cli generate --config configs/default.json --out data/sample_24h.csv
```

- The CLI commands are `generate`, `baseline`, `cooperative`, `competitive`, `compare` and `experiment`.
- `--tariffs file.csv` (columns `hour,retail_price_inr_per_kwh,export_price_inr_per_kwh`) replaces the flat tariffs.
- `experiment` takes one or more experiment IDs (`E1` … `E7`) or `all`.

## Dashboard

`python -m streamlit run app.py`. The sidebar holds every control:

- **Scenario:** the golden one-hour fixture, the committed seed-42 day, or a generated day (any seed, solar scale 0.5–2×, 4 or 6 microgrids).
- **Tariffs and fees:** retail b, export a, seller and buyer fees.
- **Mechanism parameters:** bargaining weight α and the seller-game price tick.

Profiles depend only on the scenario controls, so changing tariffs or fees never regenerates them (a test
checks this). If a solver or validation step fails, the page shows a message and no numbers.

| Tab | Shows |
|---|---|
| Overview | The model in brief, scenario facts, community net cost per mechanism, equilibrium banners |
| Profiles | Generation, demand and net position per microgrid; hourly community surplus vs deficit |
| Simulate | Parameters in use, what each control changes, the full metric table, solve times |
| Transactions | Trade log and per-interval settlement; hourly P2P volume |
| Game analysis | Equilibrium status per interval. For any interval: verified NE or **NO PURE EQUILIBRIUM FOUND** with the **NOT NASH** diagnostic, utility curves, a regret heatmap and the full payoff table |
| Comparison | Metric table, net cost per mechanism, per-microgrid gains, hourly gain, interpretation notes |
| Network | Sankey diagram of contracted seller → buyer energy |
| Downloads | Config JSON; summary, per-microgrid, trades, settlements and diagnostics as CSV and JSON |

**Chart conventions:**
- **Colours are validated:** they pass the palette checker for colour-blindness in light and dark mode.
- **One y-axis per chart.**
- **Units on every axis.**
- **Every chart has a table view.**

## Repository layout

```
microgrid-p2p-game-theory/
  README.md  requirements.txt  pyproject.toml  .gitignore  app.py
  configs/default.json  configs/golden_one_hour.json
  data/golden_one_hour.csv  data/sample_24h.csv
  src/microgrid/
    models.py  data.py  energy.py  settlement.py  baseline.py  cooperative.py  competitive.py
    metrics.py  simulation.py  experiments.py  cli.py
  scripts/make_figures.py      # report figures from saved results (matplotlib; outside the package)
  scripts/audit_exports.py     # independent export audit (does not import the package)
  tests/                       # 17 test modules, 345 tests
  docs/  report.md  experiment_protocol.md  model_specification.md  demo_script.md  audit_report.md
         report_outline.md  figures/fig01..fig12.png
  results/                     # generated outputs (git-ignored; regenerate with the commands above)
```

## Architecture

Dependencies flow one way, and only `app.py` and `scripts/` use plotting libraries (a test checks this with an
AST scan). Every mechanism decides only *which* trades happen and at *what* price. All energy flows and money
come from the single shared `settle_market`, which validates every result before returning it.

```
models <- data <- energy <- settlement <- baseline / cooperative / competitive <- simulation <- experiments <- cli
                                                                    metrics  <-+                app (dashboard)
```

| Module | Responsibility |
|---|---|
| `models.py` | Frozen, validated dataclasses; error and warning types; CSV column contracts |
| `data.py` | Config and tariff files, profile validation, seeded synthetic profiles, trade / settlement / diagnostics tables |
| `energy.py` | Surplus/deficit, roles, immutable `HourState`, energy-balance checks |
| `settlement.py` | The single shared settlement plus `validate_market` (conservation, capacities, payment cancellation, fees, community-gain identity, voluntary participation) |
| `baseline.py` | Utility-only market via `settle_market(state, trades=())` |
| `cooperative.py` | HiGHS LP (status-checked), duality-based lexicographic tie rule, greedy cross-check, bilateral Nash price |
| `competitive.py` | Price grid, buyer rule, full payoff table, regrets, all pure NE, `NO_PURE_NE` / cap reporting |
| `metrics.py` | `compute_metrics` (every blueprint metric, explicit N/A) and `jain_fairness` |
| `simulation.py` | `compare_all`: every mechanism on identical frozen states, timing, input-mutation guard, metrics against the LP optimum |
| `experiments.py` | E1–E7: one parameter family at a time, frozen or hash-tracked profiles, saved configs, deterministic tables |
| `cli.py`, `app.py` | Thin adapters, with no market maths |

## Model in brief

Microgrids `i`, hourly intervals `t`; energy in kWh per interval, money in INR, prices in INR/kWh. Costs are
signed: positive = pays. Full notation is in [docs/model_specification.md](docs/model_specification.md).

- **Baseline:** `S = max(G−D,0)`, `R = max(D−G,0)`, `C_base = b·R − a·S`.
- **Settlement:** `B = R − P2P_in`, `X = S − P2P_out`, `C = b·B − a·X + payments − receipts + fees`, `U = C_base − C`.
- **Cooperative:** maximise `Σ (b − a − f_s − f_b)·q` (HiGHS). The tie rule is applied over the exact optimal face. The price is `(1 − α)(a + f_s) + α(b − f_b)`. This is a benchmark plus bilateral bargaining, **not** a Nash equilibrium.
- **Competitive:** sellers post prices on a tick within `[a + f_s, b − f_b)`, and buyers take the cheapest offers first (ties by seller ID). The payoff is gain over exporting. A pure NE is a profile with `max_i regret_i ≤ tol`, checked over every unilateral deviation. `NO_PURE_NE` intervals are not settled.

## Key results (synthetic data; full analysis in [docs/report.md](docs/report.md))

| Reference day (seed 42) | Baseline | Cooperative | Competitive |
|---|---|---|---|
| Community net cost (INR) | 8,633.60 | 7,373.66 (−14.6%) | 7,771.52 (−10.0%) |
| P2P volume (kWh) / average price (INR/kWh) | 0 / N/A | 251.99 / 5.50 | 172.42 / 7.50 |
| Jain fairness · efficiency vs LP optimum | N/A · 0 | 0.830 · 1.000 | 0.467 · 0.684 |

- **Where competition falls short:** the competitive market trades the LP-optimal volume in every hour that has a pure equilibrium. Its whole shortfall comes from hours 11–12, oversupplied hours with no pure equilibrium.
- **Who gains:** competition hands most of the gain to sellers.
- **Fees can help competition:** moderate fees (≥ 3 INR/kWh) remove that price war, and all trade stops at fees = spread (5 INR/kWh).
- **Across ten seeds:** cooperative gain is 1,284.80 ± 30.72 INR and competitive gain 898.80 ± 124.02 INR (efficiency 0.51–0.85).

The golden fixture reproduces the blueprint: baseline 230, cooperative 80, gain 150, trades 15/5/10 kWh at
INR 5.50. Its competitive NE is (7.50, 7.50), with the same total but gains of 90/7.5/45/7.5.

## Testing and audit

**Recorded run** (2026-10-27, inside `.venv`):

```
$ python -m pytest -q
345 passed in 9.41s
$ python scripts/audit_exports.py results/experiments/E1
22 passed, 0 failed
```

**What the 17 test modules cover:**
- the blueprint fixtures and T01–T16
- settlement invariants
- bargaining optimality
- the LP core and its failure handling
- an independent reference enumeration of the competitive game
- the controller and metrics
- a seed-42 regression snapshot
- experiment design rules (one variable at a time, frozen profiles, byte-identical outputs)
- figure generation
- tampering detection by the export audit
- dashboard behaviour driven through `AppTest`

The full requirement-by-requirement verdict is in [docs/audit_report.md](docs/audit_report.md).

**Tested environment:** macOS (Darwin 25.6.0, Apple silicon), Python 3.13.9, numpy 2.5.4, pandas 2.3.3,
scipy 1.18.1, streamlit 1.66.1, plotly 6.9.0, matplotlib 3.11.2, pytest 8.4.2. CVXPY is optional and unused.

## Assumptions and limitations

- **No physics beyond energy accounting:** grid-connected, lossless, no storage, network or forecasting.
- **Hourly intervals; synthetic data; illustrative flat tariffs** (an hourly tariff table is supported).
- **Flat per-kWh fees** paid to an external operator.
- **Bilateral bargaining:** quantities are fixed centrally first.
- **Tie rules use string IDs**, for the cooperative allocation and for buyers' choice between equally priced sellers. They affect individual outcomes; this is disclosed and tested.
- **Competitive game:**
  - prices are discrete
  - `NO_PURE_NE` intervals are settled without trade (a modelling choice; mixed equilibria are not computed)
  - exact enumeration is capped at 3 sellers by default
  - it is not a Stackelberg equilibrium
- **Higher P2P volume is not an emissions claim.**
- **Zero-pad microgrid IDs** (`MG01`) beyond nine.

## Possible extensions (blueprint section 17)

- Heterogeneous tariffs and pairwise fees
- Storage
- Forecast uncertainty
- Network-constrained power flow
- Multilateral bargaining or Shapley allocation
- Learning or repeated-game agents

Each needs updated equations, data contracts and test fixtures before its results can be compared with the MVP.
