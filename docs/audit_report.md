# Independent audit report (blueprint Prompt 6)

- **Scope:** the whole repository at v0.4.0, checked against every MVP requirement, the definition of done, the critical tests T01–T16, the risk table and the final readiness checklist.
- **Date:** 2026-10-27.
- **Independence:** this audit was performed by the same assistant that wrote the code. To reduce that bias:
  - the export checks run in `scripts/audit_exports.py`, which does **not** import the `microgrid` package and re-derives everything from the exported CSVs;
  - the competitive game is cross-checked against a naive reference enumeration written separately inside the tests.

  A human or fresh-context reviewer is still recommended before submission.

## Commands executed

| Command | Result |
|---|---|
| `python -m pytest -q` | **345 passed** in 9.41 s |
| `python -m microgrid.cli compare --config configs/golden_one_hour.json --profiles data/golden_one_hour.csv` | Net cost 230.00 / 80.00 / 80.00; gain 150.00; price 5.50 / 7.50 |
| `python -m microgrid.cli experiment all --out-dir results/experiments` | E1–E7 complete, 88 files |
| `python scripts/audit_exports.py results/experiments/E1` | **22 passed, 0 failed** |
| Tampered copies of the E1 exports (shifted price, false NE label, altered cost) | Audit **fails** on each (tests `test_export_audit_detects_tampering`) |

## Prompt 6 focus items

| Item | Verdict | Evidence |
|---|---|---|
| Hidden shortcuts | PASS | See note 1 below |
| False Nash labelling | PASS | See note 2 below |
| Missing payment cancellation | PASS | `settlement.py:309` raises if Σ purchases ≠ Σ sales. The export audit checks community cost = grid bill + fees. `test_internal_payments_cancel`, `test_payments_cancel_every_hour` |
| Double-counted fees | PASS | `settlement.py:312` (fees = operator revenue) and `:319` (community gain = Σ q(b − a − f_s − f_b)). The cooperative mechanism also checks settled gain = LP welfare. Fee fixture: gain 90 with fees 60 |
| Inconsistent units | PASS | Energy is kWh per one-hour interval everywhere, with no kW anywhere; money is INR; prices are INR/kWh. Every CSV column name and figure axis carries its unit |
| Price-tie sensitivity | PASS (disclosed) | Buyer ties by seller ID are documented. `test_tie_rule_decides_outcomes_between_identical_sellers` shows an asymmetric equilibrium set for identical sellers. The cooperative long-side tie effect is documented. The README never claims independence from the tie rule |
| Ignored solver failures | PASS after fix F1 | See note 3 below |
| Baseline mutation | PASS | `simulation.py:116–129` snapshots the input and raises if any mechanism changes it (tested with a deliberately mutating solver). T16 tests in edge cases, cooperative, competitive and simulation. Baseline costs are identical across mechanisms (`test_every_mechanism_runs_on_identical_states`) |

**Notes:**

1. **Hidden shortcuts.**
   - The cooperative allocation comes from HiGHS LPs (`cooperative.py:259`). The greedy rule is only an independent cross-check (`:365–375`).
   - The competitive game builds the full payoff table and takes regrets along every axis (`competitive.py:319–322`). Nothing is sampled.
   - Unknown config keys are rejected (`models.py`, `from_dict`).
2. **False Nash labelling.**
   - `pure_ne_found=True` is set only at `competitive.py:414`, on the branch reached when exhaustive enumeration found equilibria.
   - The `NO_PURE_NE` branch (`:375–376`) settles nothing.
   - The cap branch claims nothing.
   - The export audit re-checks every settled equilibrium, and every NOT NASH profile, from the exported unilateral-deviation curves.
3. **Ignored solver failures.**
   - Every LP status is checked (`cooperative.py:260`).
   - A `SolverError` is raised on missing duals (`:337`), welfare loss (`:363`) and a greedy mismatch (`:368`, `:375`).
   - The CLI exits with code 2 and a message (`cli.py:427`).
   - **Dashboard (fixed during this audit, see F1):** it now stops with a message and shows no numbers (`app.py:673–674`; `test_solver_failure_is_reported_without_numbers`).

## MVP features (blueprint section 2)

| ID | Requirement | Verdict | Evidence |
|---|---|---|---|
| M1 | 24 hourly observations for 4 microgrids | PASS | 96 unique rows with a fixed seed of 42; `test_reproducibility.py` |
| M2 | Utility-only baseline | PASS | Golden −60 / 120 / −30 / 200 = 230; `test_baseline.py` |
| M3 | Lossless P2P allocation | PASS | Capacities checked in `validate_market`; `test_capacities_are_respected_every_hour` |
| M4 | Cooperative bargaining prices | PASS | `bargaining_price`; positive-surplus rule at `cooperative.py:141`; the closed form equals the numerical argmax (`test_bargaining.py`) |
| M5 | Discrete simultaneous seller-price game | PASS | `price_grid` and `allocate_buyers` (`competitive.py:135`) |
| M6 | Pure Nash verification | PASS | Every deviation checked; matches the reference enumeration on 5 games and 10 trading hours |
| M7 | Full financial settlement | PASS | One `settle_market`; the export audit's money checks pass |
| M8 | Comparative metrics | PASS | `compute_metrics` (`metrics.py:74`): volume, gains, costs, fairness, regret, runtime, with explicit N/A (`:113–128`) |
| M9 | pytest validations | PASS | 345 tests |
| M10 | Streamlit demonstration | PASS | Run, inspect, modify, compare and export CSV; `test_dashboard.py` and a live browser session |

## Definition of done (blueprint section 2)

| # | Item | Verdict | Note |
|---|---|---|---|
| 1 | Clean setup; section 14 commands execute | PASS with an environment caveat | See note below |
| 2 | Schemas validate with descriptive errors | PASS | Row, CSV-line and column-specific messages; `test_data_validation.py` |
| 3 | Golden fixture reproduced | PASS | Baseline 230, cooperative 80, gain 150, 30 kWh, price 5.50 |
| 4 | Quantities and money reconcile | PASS | 22/22 export checks |
| 5 | Equilibria independently checked; no-NE labelled | PASS | Reference enumeration and export curve checks; `NO_PURE_NE` with NOT NASH diagnostics |
| 6 | At least five reproducible experiments with charts | PASS | 7 experiments, 12 figures, byte-identical result tables |
| 7 | Dashboard, README, result tables, demo script | PASS | `app.py`, `README.md`, `results/experiments/*`, `docs/demo_script.md` |

**Note on item 1.** A fresh `.venv` installed from `requirements.txt` in Phase 1, and `pytest` and
`streamlit run` work. On this machine, iCloud marks `.venv` as hidden, so Python 3.13 ignores the editable
`.pth` file and `python -m microgrid.cli` needs `PYTHONPATH=src` (README troubleshooting). This is not a code
defect.

## Critical tests T01–T16 (blueprint section 11)

| Test | Covered by |
|---|---|
| T01 golden 230 / 80 / 150 | `test_golden_fixture.py::test_baseline_net_costs`, `::test_cooperative_costs_and_gains` |
| T02 allocation | `::test_cooperative_trades` |
| T03 p = 5.5 | `test_bargaining.py::test_golden_bargaining_price` |
| T04 fees 1 + 1 | `test_golden_fixture.py::test_fee_fixture`, `::test_competitive_fee_fixture` |
| T05 fees ≥ spread | `::test_trading_stops_when_fees_absorb_the_spread` (both mechanisms) |
| T06 night hour | `test_edge_cases.py::test_night_hour_has_no_sellers` |
| T07 no sellers or buyers | `test_edge_cases.py`, `test_cooperative.py`, `test_competitive_equilibrium.py` (`no_sellers_or_no_buyers`) |
| T08 seller exhausted | `test_settlement.py::test_seller_cannot_sell_more_than_its_surplus` |
| T09 payment cancellation | `test_settlement.py::test_internal_payments_cancel`; export audit |
| T10 energy conservation | `test_energy_invariants.py`; `test_cooperative.py::test_energy_identity_every_player_and_hour`; export audit |
| T11 pure NE report | `test_competitive_equilibrium.py::test_equilibria_match_independent_enumeration` (+ trading hours) |
| T12 no pure NE | `::test_price_war_has_no_pure_equilibrium_and_none_is_claimed` |
| T13 deterministic tie | `test_edge_cases.py::test_repeated_runs_are_identical`; `test_cooperative.py::test_tie_rule_is_lexicographic_among_optimal_allocations` |
| T14 seed reproducibility | `test_reproducibility.py`; `test_experiments.py::test_result_tables_are_byte_identical_on_rerun` |
| T15 bad data | `test_data_validation.py` (44 cases) |
| T16 inputs unchanged | `test_edge_cases.py::test_inputs_are_not_mutated`; `test_simulation.py`; `test_dashboard.py::test_tariff_changes_do_not_regenerate_profiles` |

## Final readiness checklist (blueprint section 20)

| Check | Verdict |
|---|---|
| Physics: no negative energy, all energy balances hold | PASS |
| Economics: no unaccounted internal cash or external fees | PASS |
| Cooperative game: positive trade surplus, derived bargaining price | PASS |
| Competitive game: fully checked seller deviations, or an honest no-NE | PASS |
| Repeatability: identical scenario and seed give identical results | PASS |
| Human demo: run the app, change parameters, view tables, export results | PASS |
| Academic report: theory matches code, results not invented | PASS (`docs/report.md` quotes only saved outputs) |
| Handover: fresh install, tests and demo documented | PASS (with the environment caveat above) |

## Findings and corrections

| ID | Severity | Finding | Action |
|---|---|---|---|
| F1 | Medium (presentation) | The dashboard would show a raw Streamlit traceback if a solver or validation error occurred | **Fixed.** `app.py` now catches validation, energy, settlement and solver errors, shows "could not produce a validated result" and stops; test added |
| F2 | Environment | iCloud-hidden `.venv` breaks the editable install for `python -m microgrid.cli` | **Documented** in the README with two workarounds; not a code defect |
| F3 | Low (documented deviation) | Some signatures differ from blueprint section 10 | **Kept and documented.** See the list below |
| F4 | Disclosure | Modelling choices that shape competitive results | **Disclosed** in the README, `model_specification.md`, `report.md` §7 and the protocol. See the list below |
| F5 | Interpretation | E5 at N = 6: 6 of 10 game hours exceed the 3-seller cap, so the competitive total is not a market outcome | **Reported as such** in the report and Fig. 8; exact results need a higher cap |

**F3 deviations from the blueprint signatures:**
- `load_and_validate_profiles(path, cfg)` takes the whole config (rather than `ids`), because validation needs the horizon too.
- `settle_market(..., mechanism=...)` requires the mechanism keyword, so a result can never be mislabelled.
- `solve_*` accept optional tariffs.
- The baseline returns `MarketResult` rather than a separate `BaselineResult`.

**F4 modelling choices:**
- `NO_PURE_NE` intervals are settled without trade.
- The buyer tie rule uses seller ID.
- Prices are discrete.
- Exact enumeration is capped at 3 sellers.
- There are no mixed equilibria.

No blocking findings remain open. After the F1 fix the suite was rerun: **345 passed**.
