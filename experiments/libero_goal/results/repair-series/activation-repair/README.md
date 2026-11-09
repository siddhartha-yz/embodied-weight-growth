# SmolVLA: localize first, then compare BP and direct writing

This pilot tests where the benefit of Astra comes from. All arms start from the same stock SmolVLA-LIBERO. Only paired Y/Z coefficients of existing action-head feature columns can change; each edited arm may use eight column pairs (16 scalars), and each scalar may move at most 0.65 from stock.

| Arm | Location selection | Values |
| --- | --- | --- |
| Stock | None | Original weights |
| Auto + BP | Deterministic selection using the shared intervention effects and the common objective | Projected Adam through the actual action sampler |
| Astra + BP | GPT-6 Astra via Codex SDK, using the same eligible column pool | Same BP procedure as Auto + BP |
| Astra + direct | Same frozen Astra selection as Astra + BP | A separate Codex SDK call assigns absolute values |

The development probes temporarily intervene on a single feature's contribution to the Y/Z readout inside the existing action sampler, holding observation and seeded noise fixed. The intervention is measured separately on approaching/grasping, ordinary carrying, and near-failure carrying. The intervention is a diagnostic tool: the scored rollout loads absolute weights before its first action and uses the unmodified inference path. It has no trigger, per-object routing, or runtime hook.

Both optimization arms use the same 32-step budget and weighted objective. Each repair group averages Y/Z MSE over its verified successful action snippets. The preservation groups use original native 7D actions, weight 5 for approach/grasp and weight 1 for ordinary transport/placement. The two repair groups each have weight 0.5. The same training targets and loss are supplied to the direct writer along with local forward-probe evidence.

The development bottle and bowl cases are unchanged, as are the eight old retention cases. New layouts use the previously unused official bottle initial-state bank: four bowl, two crossed bottle, two micro-offset bottle. Initial-state validation uses geometry only. The scorer pairs initial simulator state, both images, robot state, instruction schedule and seed. Placement requires 40 consecutive controls (2 s) on the cabinet without sustained grasp; the retention panel uses the native goal.

## Results

| Fixed from the beginning | Stock | Auto + BP | Astra + BP | Astra + direct |
| --- | ---: | ---: | ---: | ---: |
| Development bowl | Fails | Stable at 166 | Stable at 133 | Stable at 133 |
| Development bottle | Lifts correctly, fails placement | No sustained lift | No sustained lift | No sustained lift |
| Fresh bowl, stable release | 0/4 | 3/4 | 4/4 | 3/4 |
| Fresh bottle, stable release | 2/4 | 3/4 | 2/4 | 2/4 |
| Old tasks, native success | 8/8 | 7/8 | 6/8 | 6/8 |
| Common fit objective (lower is better) | .107273 | .020646 | .025428 | .024798 |

All three edits lose the two-mug episode. Both Astra-location arms also lose ordinary bottle case 0. Auto and Astra-BP each complete six of eight new cases, but their gains differ: only Astra-location arms place the bowl on new layout 14; only Auto + BP places the bottle on crossed layout 25. The direct writer does not improve upon BP on Astra's chosen positions in this sample.

### Localization: individually selective pulses do not guarantee combined preservation

The [intervention study](LOCALIZATION.md) shortlisted 24 new feature columns. The two selectors shared only four of their eight chosen columns. In exact single-context probes, adding +0.20 of feature 491 to the Y readout lowered bottle and bowl target error by 10.97% and 9.08%; the all-seven-axis RMS shift was 0.0066 at additional pregrasp times. However, in the same four extra pregrasp times, the **complete edits** shift the original actions by RMS 0.0633 / 0.0626 / 0.0628 for Auto + BP / Astra + BP / Astra + direct. These are fixed-input measurements, not claims of performance at every grasp. The development bottle and old-task rollouts confirm that none of the combinations eliminates acquisition regressions from the start.

[Paired scorecard: 18 cases, 72 rollouts](scorecard.json) · [Combined-edit checks](combined-protection.json) · [Probe results](localization-compact.json)

[New bowl layout 14](../../../media/activation-bowl-layout14.mp4) · [New bottle layout 25](../../../media/activation-bottle-layout25.mp4) · [Original bottle failure](../../../media/activation-bottle-dev.mp4) · [Development bowl](../../../media/activation-bowl-dev.mp4)

[Replay the weights or rerun the probes](../REPRODUCE.md#bp-control-and-activation-localization).
