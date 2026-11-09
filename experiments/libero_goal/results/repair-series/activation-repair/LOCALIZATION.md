# Action-readout interventions

Start from the unchanged SmolVLA checkpoint. On each of its denoising calls, temporarily add `s × 0.2 × activation[column]` to one selected Y or Z readout row, where `s` is +1 or −1. Do this at every control token without a phase or object switch. The hook is an instrument: the same contribution can be expressed by changing that one weight, and the hook is removed after each diagnostic forward. Hold the observation, checkpoint, and random action noise fixed.

The 19 cached BP-control contexts supply 3 demonstrated repair chunks and 16 stock action chunks to protect. Preservation contexts with the instructed object held are **normal carry**; those without it are **pregrasp**. Eight extra development times are replayed from the same four old trajectories for a separate check: four before grasp and four during normal carry. The eight check times are not used to fit weights and are not independent layouts. No final evaluation rollouts enter the probe.

A frozen-activation approximation nominates 24 new hidden columns (excluding the eight columns explored in the previous BP/WSP round). Actual full-sampler signed interventions measure native action effects for both rows in those columns. A batch includes an unchanged sibling with the same input/noise; subtract its actions from each intervened sibling's actions. This controls for differences in BF16 arithmetic from changing the batch shape. The eight strongest discovery pulses by mean teacher MSE improvement divided by pregrasp disturbance are also confirmed with single-context forwards on all phases and check times.

The two repair traces contribute separate target MSEs on the native Y/Z commands. Preserve diagnostics report all-seven-axis RMS, maximum perturbation, mean Y/Z changes, and per-context RMS. Readout effects at fixed observations are causal changes to model actions, not evidence of successful closed-loop placement or of unaffected grasp in every scene.

- [localization-compact.json](localization-compact.json): candidate stock coordinates, signed phase summaries, and exact single-context checks.
- [localization.json](localization.json): same metadata plus the original 19 action target/stock chunks and per-context +0.2 full-action probes.

## Measured examples

| Exact single-context pulse | Bottle teacher MSE drop | Bowl teacher MSE drop | Extra pregrasp RMS | Extra normal-carry RMS |
| --- | ---: | ---: | ---: | ---: |
| +0.20 × h491 into Y | 10.97% | 9.08% | 0.0066 | 0.0041 |
| −0.20 × h491 into Z | 9.41% | 13.11% | 0.0080 | 0.0047 |
| −0.20 × h458 into Y | 14.05% | 2.08% | 0.0034 | 0.0051 |

The stock controls reproduced the cached native actions to within 1e−6 in single-context forwards. The maximum stock batch-versus-single difference was 0.003935 native command units. Use the same-batch differences for the 48-coordinate candidate pool and exact single-context results for the representative measurements above.

## Combined locked edits

Run `combined_protection.py` on the same eight extra development contexts after all three edit manifests are locked. It applies each complete manifest from the same stock checkpoint, uses a single-context sampler with unchanged noise, and restores stock after measuring that arm. It does not tune weights or access final blind rollouts. The fixed-input stock replay error is zero.

| Locked arm | Pregrasp RMS | Pregrasp peak command difference | Normal-carry RMS | Normal-carry peak command difference |
| --- | ---: | ---: | ---: | ---: |
| Automatic localization + BP | 0.0633 | 0.2853 | 0.0255 | 0.1561 |
| Astra localization + BP | 0.0626 | 0.2279 | 0.0241 | 0.1035 |
| Astra localization + WSP | 0.0628 | 0.2391 | 0.0271 | 0.1249 |

RMS averages across all seven native command axes and ten controls per context. Peaks are maximum absolute single-axis deviations across the four relevant windows. The three arms have similar mean pregrasp perturbations, even though individual diagnostic pulses were selective. All pregrasp peaks occur at the bowl development trace's t=50; all normal-carry peaks occur at the clean bottle trace's t=60. Full action shifts and worst-window RMS are in `evidence/combined-protection.json` locally and `run-01/combined-protection.json` on the lab.
