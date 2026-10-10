# Skallagrim initial gauntlet results

Completed required pairings: 15/15. Each pairing requires 50 games, 300 ms search requests, 25 swapped-role opening pairs. The common IPC allowance is 25 ms and maximum game length is 600 plies.

These are bootstrap experiments, not a claim of federation-certified superiority. Native Python opponents have documented shieldwall mismatches and use our labeled warm-start weights. Interpret timeout/illegal-move/compatibility forfeits separately from play. Static fort detection and referee rules remain subject to federation compatibility certification.

| Candidate | Opponent | Games | W/D/L | Match score | Played score | Attacker match | Defender match | Paired 95% interval | Forfeits | CPU |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| beta | luizz | 50 | 50/0/0 | 100.0% | 100.0% (22 games) | 100.0% | 100.0% | 100.0%–100.0% | 28 | 6 |
| beta | nicholas | 50 | 47/0/3 | 94.0% | 50.0% (4 games) | 92.0% | 96.0% | 88.0%–100.0% | 46 | 6 |
| beta | opentafl | 50 | 50/0/0 | 100.0% | — (0 games) | 100.0% | 100.0% | 100.0%–100.0% | 50 | 6 |
| beta | taflzero | 50 | 50/0/0 | 100.0% | — (0 games) | 100.0% | 100.0% | 100.0%–100.0% | 50 | 6 |
| iota | beta | 50 | 32/0/18 | 64.0% | 60.6% (33 games) | 40.0% | 88.0% | 52.0%–76.0% | 17 | 5 |
| iota | luizz | 50 | 50/0/0 | 100.0% | 100.0% (25 games) | 100.0% | 100.0% | 100.0%–100.0% | 25 | 5 |
| iota | nicholas | 50 | 46/0/4 | 92.0% | 0.0% (4 games) | 84.0% | 100.0% | 84.0%–98.0% | 46 | 5 |
| iota | opentafl | 50 | 49/0/1 | 98.0% | — (0 games) | 100.0% | 96.0% | 94.0%–100.0% | 50 | 5 |
| iota | taflzero | 50 | 50/0/0 | 100.0% | — (0 games) | 100.0% | 100.0% | 100.0%–100.0% | 50 | 5 |
| rho | beta | 50 | 40/0/10 | 80.0% | 72.7% (22 games) | 88.0% | 72.0% | 66.0%–92.0% | 28 | 4 |
| rho | iota | 50 | 29/0/21 | 58.0% | 65.1% (43 games) | 32.0% | 84.0% | 46.0%–70.0% | 7 | 4 |
| rho | luizz | 50 | 49/0/1 | 98.0% | 97.2% (36 games) | 100.0% | 96.0% | 94.0%–100.0% | 14 | 4 |
| rho | nicholas | 50 | 49/0/1 | 98.0% | 95.8% (24 games) | 96.0% | 100.0% | 94.0%–100.0% | 26 | 4 |
| rho | opentafl | 50 | 48/0/2 | 96.0% | — (0 games) | 96.0% | 96.0% | 90.0%–100.0% | 50 | 4 |
| rho | taflzero | 50 | 50/0/0 | 100.0% | — (0 games) | 100.0% | 100.0% | 100.0%–100.0% | 50 | 4 |

Role scores conditional on completing play without a protocol failure:

| Candidate | Opponent | Attacker played | Defender played |
| --- | --- | ---: | ---: |
| beta | luizz | 100.0% (22 games) | — (0 games) |
| beta | nicholas | 33.3% (3 games) | 100.0% (1 games) |
| beta | opentafl | — (0 games) | — (0 games) |
| beta | taflzero | — (0 games) | — (0 games) |
| iota | beta | 35.0% (20 games) | 100.0% (13 games) |
| iota | luizz | 100.0% (21 games) | 100.0% (4 games) |
| iota | nicholas | 0.0% (4 games) | — (0 games) |
| iota | opentafl | — (0 games) | — (0 games) |
| iota | taflzero | — (0 games) | — (0 games) |
| rho | beta | 0.0% (2 games) | 80.0% (20 games) |
| rho | iota | 38.9% (18 games) | 84.0% (25 games) |
| rho | luizz | 100.0% (24 games) | 91.7% (12 games) |
| rho | nicholas | 95.7% (23 games) | 100.0% (1 games) |
| rho | opentafl | — (0 games) | — (0 games) |
| rho | taflzero | — (0 games) | — (0 games) |

Match scores include protocol forfeits; the last column before CPU counts those forfeits. Played score is conditional on a game finishing without a protocol failure and can be selection-biased; it is not an independent strength estimate. Full termination breakdowns and side results are preserved in JSON summaries. Confidence intervals resample opening pairs, not individual games.

Hardware: Intel Core i7-1255U, 16 GiB RAM. Both players in each match are pinned to one equivalent Windows efficiency core. CPU-only inference; single-thread PyTorch/ONNX; Java heap 256 MiB. Three concurrent campaigns use logical CPUs 4, 5 and 6. Ambient desktop scheduling and package power limits remain uncontrolled.

Incomplete pairings: none

Every pairing has immutable artifact hashes, model provenance, opening history, process stderr and complete move logs under the supplied run root.
