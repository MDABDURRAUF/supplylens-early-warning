# SupplyLens Early Warning

**A supply chain early-warning system: it watches daily signals from suppliers, ports, regions and commodity
prices, and warns while a problem is still building.**

Every day, for every supplier, port, region and commodity, it gives:

1. **A risk index from 0 to 100** with a level (Watch, Warning or Critical), built from how far each signal has
   moved from that entity's own normal range.
2. **The reason and the stakes**: which signals moved, which SKUs and how much yearly spend sit behind the
   entity, and a first step to consider.

Everything is computed "as of" a date from data up to that date only, so any past day can be replayed. The
alerts are scored against disruptions whose dates are known, next to a plain fixed-threshold rule, EWMA and
CUSUM detectors, and a machine-learning model that estimates whether a disruption will begin within 14 days.

The included data is simulated. Results on it show how the method behaves on such data; they are not findings
about real suppliers, ports or markets.

## What it does

| Question | Where | Function |
|---|---|---|
| Who is in alert today, how serious is it, and what should I look at first? | **Today's alerts** | `build_alerts`, `current_alerts` |
| What would it have said on an earlier day? | **Today's alerts**, as-of date | `detect(signals, as_of=...)` |
| Why is this entity in alert? | **Entity explorer** | `explain`, `robust_baseline` |
| What is exposed: which suppliers, SKUs and spend? | **Exposure** | `exposure_table`, `exposed_skus` |
| Who is not in alert yet but likely to start? | **Today's alerts**, watch list | `OnsetModel`, `model_watchlist` |
| How early and how reliable are the warnings, and against what? | **Detection performance** | `evaluate.compare_methods`, `by_level`, `by_event_type` |
| How much would the result change on another history? | command line, `--seeds` | `evaluate.across_seeds` |

Outputs: a daily alert digest and an evaluation report (single-file HTML), the alerts and the method comparison
as CSV, and a command line for all of it.

## Results on the simulated data

40 suppliers in US states, 5 port gateways, the 4 Census regions and 3 commodity price indices: 52 entities
reporting 212 daily series for 730 days (1 July 2024 to 30 June 2026). 77 disruptions are planted with a known
onset, peak and end: 36 supplier capacity problems, 18 quality drifts, 10 regional weather events, 9 port
congestions and 4 commodity price spikes. Each builds up gradually. 150 SKUs carry $41,438,100 of yearly spend.

A disruption counts as detected when the entity is in alert between 14 days before the onset and the **peak**;
a warning after the worst day does not count. Detectors on the whole history after the baseline's burn-in
(28 October 2024 to 30 June 2026, 77 disruptions):

| Method | Detected by the peak | Median days from onset to first alert | False alarms a month | Precision |
|---|---|---|---|---|
| Static threshold | 69% | 13.0 | 2.9 | 66% |
| EWMA | 57% | 17.0 | 2.3 | 61% |
| CUSUM | 64% | 16.0 | 1.9 | 65% |
| **Risk index, Watch or higher** | **81%** | **11.5** | 4.0 | 49% |

- On the 46 disruptions that both the risk index and the static rule caught, the index's alert came a median
  of 1.5 days earlier. The index buys its extra detections with more false alarms (4.0 a month against 2.9
  across all 52 entities).
- The alert levels trade recall for precision. Watch or higher: 81% of disruptions, 49% of alerts true.
  Warning or higher: 74% and 70%, with 1.7 false alarms a month. Critical: 52% and 82%, with 0.7.
- The gain is where a fixed limit is weakest. Quality drifts (slow, one signal): the index caught 18 of 18 by
  their peak, the static rule 10 of 18. Capacity problems (several signals at once): 92% against 83%. Port
  congestion: 78% against 56%.
- The static rule is better on abrupt events in slow-moving signals: weather events 40% against 20%, and all 4
  price spikes against 2 of 4.
- Later data cannot change earlier alerts: a test rewrites every value after a cut-off date and checks that
  each score, index and level up to that date is identical.

The onset model is trained on days before 29 September 2025 (41 onsets), calibrated on the next four months
(14 onsets), and tested from 29 January 2026 (22 onsets). All methods on that test period:

| Method | Detected by the peak | Median days from onset to first alert | False alarms a month | Precision |
|---|---|---|---|---|
| Static threshold | 64% | 22.5 | 2.6 | 75% |
| EWMA | 50% | 23.0 | 1.8 | 72% |
| CUSUM | 59% | 24.0 | 1.6 | 74% |
| Risk index, Watch or higher | 77% | 17.0 | 2.4 | 71% |
| Onset model | 27% | 2.0 | 1.6 | 47% |

- On the 11 test-period disruptions that both the index and the static rule caught, the index was a median of
  10 days earlier.
- The onset model flags few disruptions (6 of 22) but flags them early. As a predictor of "a disruption begins
  within 14 days" its ROC AUC is 0.618, against 0.541 for using the risk index as the score; its Brier score is
  0.0468 against 0.0485 for always quoting the base rate. That is a modest edge, and its probabilities are only
  roughly calibrated on this history.

One history with 22 test disruptions is thin evidence, so the same comparison was repeated on 10 other
simulated histories (seeds 1 to 10), each scored on its own test period. Averages:

| Method | Detected by the peak | Median days from onset to first alert | Days gained on the static rule | False alarms a month | Precision |
|---|---|---|---|---|---|
| Static threshold | 59% | 10.4 | | 2.8 | 68% |
| EWMA | 57% | 11.1 | +4.1 | 1.6 | 71% |
| CUSUM | 59% | 11.7 | +5.5 | 1.3 | 74% |
| Risk index, Watch or higher | 70% | 7.2 | +7.0 | 2.7 | 60% |
| Onset model | 41% | -1.6 | +15.8 | 3.5 | 48% |

"Days gained" is the median, over disruptions that both a method and the static rule caught, of how much
earlier the method's alert came. A negative number of days from onset means the alert came before the onset.

These numbers come from simulated data whose disruptions and early signs were put there by the simulation, and
the detector settings were chosen on that data. Expect different numbers on real data, and measure them against
your own record of past disruptions.

## Quick start

```bash
git clone https://github.com/MDABDURRAUF/supplylens-early-warning.git
cd supplylens-early-warning
pip install -e ".[app,dev]"
streamlit run app.py
```

Command line:

```bash
early-warning sample --out data
early-warning detect --data data --as-of 2026-03-10 --out alerts.csv
early-warning digest --data data --out digest.html
early-warning explain --data data --entity SUP-16
early-warning evaluate --data data --report evaluation.html --seeds 10
```

`--data` is a folder with `signals.csv` and, optionally, `entities.csv`, `skus.csv` and `events.csv`. With
`--as-of`, signals after that date and disruptions that had not yet begun are ignored, and the onset model is
fitted only on what was known that day.

Python:

```python
import early_warning as ew
from early_warning import evaluate

world = ew.generate_sample_world()                      # or ew.load_world("data")
det = ew.detect(world.signals)                          # baselines, EWMA, CUSUM and risk index for every day
alerts = ew.build_alerts(world, det)                    # today's alerts with exposure and a suggested action
print(alerts[["entity", "level", "risk_index", "likely_cause", "annual_spend_exposed"]])
print(ew.explain(det, alerts["entity"].iloc[0]))        # why the first one is in alert

earlier = ew.detect(world.signals, as_of="2026-03-10")  # uses nothing after that day
print(ew.current_alerts(earlier))

print(evaluate.compare_methods(det, world.events, world.entities))   # against the known disruptions
model = ew.OnsetModel().fit(world, det)                 # will a disruption begin within 14 days?
print(evaluate.model_summary(model))

ew.write_html(ew.build_digest(alerts, world, det.as_of), "digest.html")
```

Sample outputs are in [examples](examples): the alert digest, the alerts CSV, the evaluation report, the
method comparison, and the simulated entities, SKUs, events and one month of signals.

## Input data

`signals.csv` is required: one row per date, entity and signal.

| Column | Required | Meaning |
|---|---|---|
| `date` | yes | Day the value refers to |
| `entity` | yes | Supplier, port, region or commodity identifier |
| `signal` | yes | One of the signal names below |
| `value` | yes | The reading; blank means not reported that day |

| Signal | Reported by | Worse when |
|---|---|---|
| `lead_time_days`, `ack_delay_hours`, `reject_rate` | supplier | higher |
| `on_time_rate`, `fill_rate` | supplier | lower |
| `dwell_days` | port | higher |
| `weather_severity` (0 to 1) | region | higher |
| `price_index` | commodity | higher |

Three optional files add to it:

| File | Columns | Adds |
|---|---|---|
| `entities.csv` | `entity`, and optionally `type`, `name`, `state`, `region`, `port`, `commodity`, `category` | Names, Census regions, and the port, region and commodity each supplier depends on |
| `skus.csv` | `sku`, `entity` (supplier), `annual_spend_usd`, optional `description` | Exposure and ranking by spend |
| `events.csv` | `entity`, `type`, `onset`, `end`, optional `peak`, `severity`, `event_id` | Scoring against known disruptions, and the onset model |

Files are validated on load; problems are reported with row counts, and nothing is dropped silently. A signal
needs about 12 weeks of history before it gets a score, alerts are scored from day 120, and the onset model
needs at least 200 further days that include past disruptions.

## How it works

**Baseline.** For each signal of each entity, the expected value for a day is the median of its previous 91
days plus a weekday adjustment (the median gap of that weekday over the previous 26 weeks). The typical scatter
is 1.4826 times the median absolute gap between actual and expected over the previous 91 days. Medians keep a
glitch or the start of a disruption from moving the baseline. The deviation score is the gap divided by the
scatter, signed so that positive always means worse. US federal holidays and the day after are skipped.

**Persistence.** Some signals move in runs: one bad day makes the next more likely. The scatter of 7-day
averages is compared with what unrelated days would give, and scores of persistent signals are scaled down
before they are accumulated, so that an ordinary run of bad days is not read as a shift.

**Detectors.** The static threshold is a fixed limit from the first 90 days (median plus three scatters in the
bad direction), broken two days in a row. EWMA is an exponentially weighted average of the scores (weight 0.2)
with a limit of 3.5 of its own standard deviations. CUSUM adds up scores above an allowance of 0.5 and alarms
at 7.

**Risk index.** For each signal, the evidence is how close EWMA, CUSUM or a single day's jump is to its limit:
nothing below half way, 0.5 at the limit, 1 at one and a half times the limit. The same is computed for all of
an entity's signals added together, which picks up several signals each leaning slightly the wrong way. The
index is `100 x (1 - product of (1 - evidence))`, so one signal at its limit gives 50. The index must hold for
two days in a row: Watch from 40, Warning from 60, Critical from 85.

**No leakage.** Every figure for a day uses that day and earlier days only. Running on data cut off at a date
gives the same history as running on all of it, which is tested.

**Onset model.** Gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`) predict whether a
disruption begins at an entity within the next 14 days. Inputs are the 7-day level, the trend and the EWMA
score of each signal, the entity's index, and the recent index of the port, region and commodity it depends on.
The history is split by date into training, calibration (a logistic fit on the raw score) and test periods; a
day is used for training only if its 14-day outcome was known when training ended. The model raises its own
alert at the probability reached by the top 2% of entity-days in the calibration period.

**Scoring.** An entity that alerts again within 7 days continues the same alert. An alert is true if it starts
between 14 days before an onset and the end of that disruption. Otherwise it is a false alarm, unless a port or
region the entity depends on is disrupted at the time; those related alerts are left out of precision and of
the false-alarm count. Detection requires an alert by the peak, and days before the onset count only if the
alert started inside the 14-day window.

**Exposure and actions.** Suppliers are linked to SKUs and yearly spend; a port, region or commodity is linked
to every supplier that depends on it. Alerts are ranked by level, then by the spend behind them. The likely
cause is a rule of thumb from the entity type and its strongest signal, and each cause and level has a
suggested first step.

## Limits

- The sample data is simulated. Its disruptions, their early signs and the way trouble spreads from ports and
  regions to suppliers exist because the simulation puts them there.
- Detector settings were chosen on the simulated data that is scored. The static rule could be made quieter or
  more sensitive by moving its limit, and the comparison would shift with it.
- Each rate rests on a small number of disruptions (77 in all, 22 in the test period), and results differ
  between simulated histories, as the 10-history table shows.
- The risk index raised more false alarms than the simpler rules on the main history, and did worse than the
  static rule on abrupt weather and price events.
- The onset model's edge is modest, and it can only learn patterns that preceded disruptions in its history.
  Its probabilities are estimates and only roughly calibrated.
- A rolling baseline absorbs a change that lasts for months and stops alerting on it; it follows strong
  seasonal swings with a lag, which can raise false alarms.
- Pooling an entity's signals assumes their day-to-day noise is unrelated; signals that move together for
  ordinary reasons would raise false alarms.
- Onsets in a real events table are judgments made afterwards, and the scores depend on them.
- Exposure is the spend behind an entity, not expected loss. Suggested actions are rule-based starting points.

## Tests

```bash
pytest
```

Twenty-five tests: the baseline, EWMA, CUSUM, static rule, event shape, labels, scoring rules and exposure
checked against hand-worked examples; no leakage from later data; holidays and glitches; the risk index
formula; detection against ground truth; the time-based model split; alert ranking; reports; and the command
line end to end.

## Related projects

- [SupplyLens](https://github.com/MDABDURRAUF/supplylens): supply chain disruption analytics on purchase orders
- [SupplyLens Control Tower](https://github.com/MDABDURRAUF/supplylens-control-tower): track decisions and outcomes
- [SupplyLens SME Sourcing Planner](https://github.com/MDABDURRAUF/supplylens-sme-sourcing): landed-cost sourcing for small businesses
- [SupplyLens Network Simulator](https://github.com/MDABDURRAUF/supplylens-network-simulator): disruption scenarios for service and goods networks

## Author

Md Abdur Rauf — ORCID [0000-0002-5105-9892](https://orcid.org/0000-0002-5105-9892)

## License

MIT. See [LICENSE](LICENSE).
