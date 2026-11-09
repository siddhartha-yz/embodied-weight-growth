# SmolVLA: do two fixed weight repairs transfer and compose?

This lab experiment reuses the parent stage's pinned `lerobot/smolvla_libero` checkpoint and native LIBERO task implementation. Compare stock, the existing four-coefficient bottle patch (A), a separately written bowl patch (B), and the same-base additive merge A+B. All weights are loaded **before the initial observation is used** and remain fixed to the end. Neither object ground truth nor the scoring status is a policy input.

## Cases and scoring

- **Bowl development:** same official init row 2 and native instruction; only the bowl is moved +7 cm in world Y before the normal physical settling. No actuator or contact parameters change. Stock first lifts the bowl at 87 and hits the cabinet rim at 118, while still commanding closed, then loses it and fails. A-only first lifts the bowl at 50, loses it after 73 without reaching the cabinet, and also fails.
- **Bottle history:** old crossed init row 2, with the same preset instruction schedule for every arm (pick-then-place phrasing through control 80, then the native instruction). This preserves the historical failure setup, not the old midpoint patch installation. The new A-only arm starts patched and does not acquire the bottle on this case; it is not a replication of the earlier late-install repair.
- **Preservation:** seven original-layout bowl/bottle rollouts and the unchanged LIBERO-Long two-mug placement. Every stock run succeeds natively. The two-mug episode uses the established LeRobot wrapper and ends at native success.
- **Held-out:** three shifted bowl layouts and three bottle layouts, all new initial rows. Two bottle cases inherit the historical instruction schedule; the small single-bottle offset uses the original instruction throughout. Initial support and absence of collisions or a pre-achieved goal were checked without running any policy. Two unsafe bowl resets were replaced by the first two geometry-valid reserves. No held-out trace or score was used to write or select B.

Native success is the first positive official predicate within budget. For paired cabinet cases, the same policy continues for 40 additional controls; the stricter retention check requires 40 consecutive native-positive controls while the target is not held. This distinguishes a momentary rim contact from an object that stays in place. First sustained lift means two-finger contact and a height gain of at least 2.5 cm for three consecutive controls.

## Results: the bottle edit helps the bowl, but the merge loses skills

All 64 scored rollouts use the same checkpoint and matched starting observations within each layout.

| Test | Stock | Bottle edit A | Bowl edit B | A+B |
| --- | --- | --- | --- | --- |
| Development bowl | Rim collision, drops | Drops before cabinet | Native success at 108, then drops | Native success at 89, stable release at 133 |
| Historical bottle | Lifts, fails placement | No sustained lift | No sustained lift | No sustained lift |
| Three held-out bowl layouts, stable release | 0/3 | 2/3 | 1/3 | 2/3 |
| Three held-out bottle layouts, stable release | 1/3 | 0/3 | 1/3 | 0/3 |
| Eight originally successful layouts, native success | 8/8 | 6/8 | 8/8 | 5/8 |

The bottle edit rescues two held-out bowl layouts despite failing the bowl development case. The merge is the only arm to leave the development bowl on the cabinet, but it fails the bottle layout that stock can complete and loses three original successes. B alone preserves all eight native successes on this panel. The key next question is whether an edit can change the carrying behavior without changing grasp acquisition when installed from the start.

[Four-arm development video](../../../media/multitask-bowl.mp4) · [Four-arm held-out bowl video](../../../media/multitask-heldout-bowl.mp4) · [Four-arm historical bottle video](../../../media/multitask-bottle.mp4)

[Scorecard](scorecard.json) · [Per-layout results](verified-matrix.json). Two cases were still native-positive when the initial evaluation ended but needed additional unheld time: [merged development bowl](retention-main.json) and [A on held-out bowl 7](retention-heldout.json). Both were continued with the same fixed weights and verified identical replay prefixes. The per-layout file contains the original windows; the scorecard includes these verified continuations.

## Bowl edit and merge

The original action-head probes replay exact already-recorded closed-loop states and random draws, verify stock actions at six matched contexts, and measure the effect of changing each selected scalar. Codex SDK calls GPT-6 Astra, without training or edits outside the existing FP32 action-output matrix. B is relative to the original model, not to A. A first candidate reached the native predicate transiently but dropped the bowl, so the editor received one development-only trajectory and proposed a replacement. Candidate choice ranks stable bowl placement, native bowl placement, and preservation on the known original-layout bowl; no held-out results enter selection. The merge adds relative deltas, including overlaps, without dropping or clipping terms.

Each rollout has its original MP4, full actions and native telemetry, and initial simulator and public observation arrays. Comparisons require identical initial state and two camera images, identical seed and instruction schedule, and 10-control replanning across all arms. An initial unrelated-task renderer corrupted later views in a discarded A run; the isolated rerun passed exact paired reset checks. Only its matched results enter the matrix.
