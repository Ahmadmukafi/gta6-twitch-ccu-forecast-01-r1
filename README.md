# gta6-twitch-ccu-forecast-01-r1

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Ahmadmukafi/gta6-twitch-ccu-forecast-01-r1/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Ahmadmukafi/gta6-twitch-ccu-forecast-01-r1/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A scale-aware launch-shape forecaster for the make.fun GTA VI Twitch CCU bounty (round 1). It forecasts hours 2–24 of the release day from the hour-1 print, using the comparable launch curves the case itself carries.

The key finding: a launch day's shape depends overwhelmingly on how big hour 1 already is. A launch that opens huge dips mid-day and only recovers to about its hour-1 level in the evening; one that opens small grows several-fold. So the forecaster conditions the whole shape on ln(hour-1 print) and blends two estimators per hour — a quadratic regression of the log-ratio on ln(hour-1 print) with empirical residual quantiles, and a Gaussian-kernel weighted empirical quantile by similarity in ln(hour-1 print) — averaged in log space, with the quadratic's scale input winsorized so an unprecedented GTA VI hour 1 cannot extrapolate the parabola.

Leave-one-out backtest over the kit's 37 usable comparable launches (pinball loss on the log scale, the competition's own scoring): **0.377** vs the house baseline's **0.930** on the same launches (rolling-midnight launches 0.284 vs 0.393; the nine largest hour-1 launches 0.202 vs 0.253). No network, no clock, no randomness: the same case always produces the same forecast.

## Run it

```sh
node main.ts
```

Reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `/output/<id>.out` (else `$MAKEFUN_OUTPUT_DIR`).

---

<sub>Entered on [make.fun](https://make.fun/?ref=6z0bn4hz&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
