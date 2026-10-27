# Experiment protocol (E1–E7)

This protocol implements blueprint section 13. Every number in the report comes from the files this protocol
produces. Nothing is typed in by hand.

## Reproduce everything

```bash
python -m microgrid.cli experiment all --out-dir results/experiments     # about 2 s on the tested machine
python scripts/make_figures.py --results results/experiments --out docs/figures
python scripts/audit_exports.py results/experiments/E1                    # independent export audit
```

You can also run a single experiment, for example `python -m microgrid.cli experiment E3 --out-dir results/experiments`.

## Common setup (base scenario)

| Item | Value |
|---|---|
| Base config | `configs/default.json` (`four_microgrids_default`) |
| Frozen profiles | `data/sample_24h.csv`: seed 42, 96 rows, 4 microgrids, 24 hourly intervals |
| Profile fingerprint | SHA-256 of the validated table, recorded for every run in `runs.csv` and `experiment.json` |
| Tariffs | retail b = 8, export a = 3 INR/kWh (flat, illustrative) |
| Fees | seller f_s = buyer f_b = 0 INR/kWh |
| Bargaining weight | α = 0.5 (cooperative) |
| Price tick | 0.5 INR/kWh (competitive) |
| Exact-enumeration cap | 3 sellers (`max_competitive_sellers`) |
| Tolerance | 1e-8 |
| Mechanisms | baseline, cooperative, competitive, all run through `simulation.compare_all` on identical states |

## Design rules

1. **One parameter family per experiment.** Every other config field keeps its base value (checked by `test_each_experiment_varies_exactly_one_parameter_family`).
2. **Frozen profiles.** E1, E2, E3 and E6 use the identical committed table; the hashes are checked to be equal. E4, E5 and E7 vary a *generator* parameter, so each run regenerates the profiles from the base config with only that parameter changed. E7's seed-42 run reproduces the committed table hash exactly.
3. **Saved configs.** Each run's exact config is saved as `configs/<run_id>.json`.
4. **Deterministic results.** The result tables are byte-identical on re-run (`test_result_tables_are_byte_identical_on_rerun`). Wall-clock runtimes are not reproducible, so they live only in `runtimes.csv`.
5. **Every seed reported.** Stochastic variation (E7) is reported per seed plus mean, standard deviation (n − 1) and range. No seed is singled out.
6. **Figures from saved files only.** `scripts/make_figures.py` reads the CSVs and recomputes nothing.

## Experiment matrix

| ID | Question | Varied parameter (values) | Profiles | Figures |
|---|---|---|---|---|
| E1 | How do the three mechanisms differ on the same day? | none (reference run) | frozen | 1–6 |
| E2 | How does welfare respond to the tariff spread? | spread b − a ∈ {0, 1, …, 7} INR/kWh, with a = 3 fixed and b = 3 + spread | frozen | 9 |
| E3 | Where do fees stop trading? | f_s + f_b ∈ {0, 0.5, …, 5.5} INR/kWh, split equally | frozen | 7 |
| E4 | How do surplus, volume and cost change with solar output? | solar scale ∈ {0.5, 1.0, 1.5, 2.0} | regenerated (seed 42) | 10 |
| E5 | How do runtime, equilibrium existence and equity scale with market size? | microgrids N ∈ {2, 4, 6} | regenerated (seed 42) | 8 |
| E6 | How sensitive are equilibria to the strategy resolution? | price tick ∈ {1.0, 0.5, 0.25} INR/kWh | frozen | 11 |
| E7 | How much do results vary across synthetic days? | seed ∈ {42, …, 51} | regenerated | 12 |

## Output files (`results/experiments/<ID>/`)

| File | Content |
|---|---|
| `runs.csv` | One row per run and mechanism: the varied value, total surplus and deficit (kWh), profile hash, and every metric except runtime |
| `players.csv` | Per run, mechanism and microgrid: P2P sold/bought, grid import/export, baseline cost, net cost, gain |
| `intervals.csv` | Per run, mechanism and hour: status, P2P volume, gain, net cost, equilibrium count, profiles examined, regret |
| `summary.csv` | Per mechanism and key metric: n, mean, std, min, max across runs |
| `runtimes.csv` | Wall-clock solve time per run and mechanism (environment-dependent) |
| `configs/*.json`, `experiment.json` | Exact configs; a manifest with the question, values, hashes, base config and environment |
| E1 only | `profiles.csv`, `trades.csv`, `settlement.csv`, `diagnostics.csv` (blueprint record), `payoff_curves.csv` (unilateral-deviation utilities) |

## Methods

- **Baseline:** `B = R`, `X = S`, `C = b·R − a·S`.
- **Cooperative:**
  - HiGHS LP maximising Σ (b − a − f_s − f_b)·q under capacity limits.
  - Lexicographic tie rule over the exact optimal face.
  - Greedy cross-check.
  - Bilateral Nash price `(1 − α)(a + f_s) + α(b − f_b)`.
- **Competitive:**
  - Every seller posts a price on the tick within `[a + f_s, b − f_b)`.
  - Deterministic buyer rule.
  - Full payoff table with every unilateral deviation checked.
  - Settlement:
    - **verified pure NE:** the selected NE (lowest price sum, then lexicographic) is settled
    - **no pure NE:** no P2P trade is settled, and a NOT NASH diagnostic is recorded
    - **more than 3 sellers:** not enumerated and not settled (`ENUMERATION_CAP_EXCEEDED`)
- **Metrics:** blueprint section 9, with explicit N/A for zero denominators. Efficiency = gain / cooperative LP optimum.

## Interpretation rules

These follow blueprint section 13, plus three rules specific to this model.

1. **Prices redistribute gain.** At fixed quantities and fees, changing the price moves gain between players without changing the total.
2. **Equal totals are not the same outcome.** When the competitive equilibrium trades the same quantity as the cooperative LP, total gains coincide. Compare the distribution instead (per-player gains, Jain fairness).
3. **Volume is not emissions.** More P2P volume is not more renewable generation or less carbon; the model has no emissions or dispatch.
4. **Percentages need a positive baseline.** Cost reduction % is shown only when the baseline cost is positive; otherwise read the absolute INR gain.
5. **Missing equilibria are counted at baseline.** A `NO_PURE_NE` interval adds zero P2P gain to the competitive totals. Competitive totals should therefore always be read together with the count of `NO_PURE_NE` intervals.
6. **Capped hours are not observations.** `ENUMERATION_CAP_EXCEEDED` intervals are not market outcomes; they are excluded from any claim about competitive behaviour.
7. **Runtimes are environment-dependent** and are only comparable within one run of the protocol on one machine.
8. **The data are synthetic.** All profiles and tariffs are synthetic or illustrative, so findings describe the model's behaviour, not any real community.
