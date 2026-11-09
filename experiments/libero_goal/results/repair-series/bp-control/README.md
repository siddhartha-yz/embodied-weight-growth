# SmolVLA: BP versus Astra under the same repair targets

Both editors start from the same stock SmolVLA-LIBERO. Only the 16 coordinates used by the previous sequential study in the existing FP32 action-output matrix are eligible; the previous values are not reused. Each edit is bounded to ±0.65 from stock. Weights are fixed from the beginning of each rollout. No controller, trigger, or teacher runs at evaluation time.

## Shared supervision

Replay exactly up to the original correct grasp, then search short Y/Z action changes in the same physics. The wine bottle teacher changes two consecutive 10-control chunks beginning at t=90 and t=100; it first meets the native predicate at step 103 and is verified stably released at 146. The bowl teacher changes one 10-control chunk at t=110; its corresponding steps are 123 and 162. The teacher traces resume recorded stock actions outside the edited windows; these are counterfactual successes, not learned policies. Initial observations and the pre-intervention physical trajectories match stock. See [teacher trajectories and final frames](teacher-evidence/README.md).

The common objective is the mean of bottle and bowl repair MSE (Y/Z axes only), plus preservation MSE (all seven native action axes). Targets for preservation are the original actions from developmental approach/grasp and two ordinary stock-successful cases, including their placement phase. Every target is paired with the recorded observation and seeded action noise. BP differentiates through action denoising, not through simulation. Astra receives the same action references, the same objective and bounds, and a local quadratic approximation computed from measured single-coordinate forward probes; it directly proposes absolute scalar values in one Codex SDK call.

## Evaluation

Compare the original, BP and Astra edits in both development cases, eight original success cases (two also supply preservation examples), and eight geometry-locked fresh cases, four per object. Each comparison uses the same starting simulator state, both cameras, robot state, prompt schedule, and random seed. The familiar unmodified tasks are scored by native success; object-placement development and blind layouts additionally require the native predicate to stay true for 40 consecutive 20 Hz controls while the object is not in a two-finger grasp.

## Results: both repair transport, but the bottle fails earlier from a cold start

| Test | Stock | BP | Astra WSP |
| --- | ---: | ---: | ---: |
| Shared action loss (lower is better) | 0.10727 | 0.01079 | 0.01090 |
| Development bowl, from start | Fails | Stable at 135 | Stable at 134 |
| Development bottle, from start | Lifts bottle, fails placement | No sustained lift | No sustained lift |
| Fresh bowl, stable release | 0/4 | 4/4 | 4/4 |
| Fresh bottle, stable release | 3/4 | 2/4 | 2/4 |
| Originally successful cases, native success | 8/8 | 7/8 | 6/8 |

The baseline's single native success on a fresh bowl was transient; its stable-release score is 0/4. Both edits lose the same crossed fresh bottle (layout 41), where stock lifts the target. Both lose the two-mug episode, and WSP additionally loses ordinary bottle case 3. BP used 32 projected Adam steps (228 s including fitting diagnostics); the single WSP SDK call took 180 s. Neither includes common data/probe preparation.

### Hold the starting grasp fixed

A separate development-only diagnostic replays the verified stock prefix until the object is already held (after step 90 for bottle, 110 for bowl). Then each actor controls all subsequent steps from the same physical state, instruction and seeded replan noise; no teacher actions are supplied after takeover. Stock fails both. BP and WSP both place both objects stably: bottle at step 145 for both, bowl at step 163 for BP and 161 for WSP. This separates a successful local transport correction from the acquisition interference introduced when the same parameters are active from step one.

[Per-case scorecard](scorecard.json) covers 18 matched cases and 54 rollouts (44 new and 10 reused stock rollouts). [Fit metrics and BP trace](bp.json) · [WSP fit](wsp-fit.json) · [Matched-grasp diagnostic](postgrasp-summary.json).

[Development bowl, from start](../../../media/bp-bowl.mp4) · [Development bottle, from start](../../../media/bp-bottle-dev.mp4) · [Bottle with a common grasp](../../../media/bp-bottle-postgrasp.mp4).

[Replay the weights or regenerate training inputs](../REPRODUCE.md#bp-control-and-activation-localization).
