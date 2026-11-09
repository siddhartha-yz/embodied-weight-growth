# Direct SmolVLA weight repair: bottle transport after a matched grasp

The pinned official SmolVLA-LIBERO checkpoint and native LIBERO-Goal scores are the same as the parent screen. This pilot isolates moving a correctly grasped bottle onto the cabinet; it does not ask the model to redo household task planning.

- Crossed layout, init 002 / seed 1002: all arms use the same pick-then-place wording for the first 80 controls, then switch to the same official instruction. Stock acquires the bottle at step 70, stalls at the cabinet rim, opens at 106 and loses two-pad contact at 107. Both the reset and the first 80 controls must match exactly.
- The hold-only action intervention retains a close command but loses contact at step 113 and fails. The carry diagnostic holds closed, adds clearance and moves farther inward; the native goal first holds at 112. Those time-conditioned clamps are causal diagnostics, not patches or inputs to the patched controller.
- Single-coefficient forward probes use the same image/state, instruction and diffusion noise at each stock replan. Probe only the existing FP32 `model.action_out_proj` (32 × 720); the response is measured in native action space. Astra receives the probe effects and numerical trajectories, and supplies direct deltas. There is no BP, additional network, online Python policy or action override in this arm.
- Install once before control 81 in the target rollout, then leave the patch fixed. Separately install before control 1 on original-layout cases to test side effects. Original-layout init 0 bottle and init 1 bowl were not in the evidence sent to Astra.

| Controller after matched grasp | Native target result | Original-layout successes (patch active from first action) |
| --- | --- | --- |
| Original checkpoint | No goal in 300 steps | 5/5 |
| Conservative Astra edit: four existing weights | First native goal at step 116 | 4/5 |
| Assertive Astra edit: four weights and two biases | First native goal at step 101 | 1/5 |

The conservative edit changes output rows 1 and 2 (Y and Z) in hidden columns 89 and 371. It leaves the gripper head untouched. In the target run the commanded opening stays at step 106, but the changed path lets the released bottle reach the cabinet. The original-layout failures under the strong edit are lost acquisitions, not merely later placements. The conservative patch also loses the held-out bottle on init 0 while preserving the held-out bowl on init 1. This is one repaired development case, not evidence of transfer to new crossed layouts.

- [Both numerical proposals](astra.json); [conservative exact tensor changes](conservative/patch.json) and [episodes](conservative/episodes.json); [assertive exact changes](assertive/patch.json) and [episodes](assertive/episodes.json).
- [Actual-video comparison](../../../media/repair-bottle.mp4). The native-goal clips stop at the first positive simulator score. The separate continuous settling check exactly reproduced all 116 controls, then ran the same patched policy for 40 more. The bottle remained ungrasped and on the cabinet for all 40 controls; it settled lying on its side. [Telemetry](settling.json).
- [Replay the locked repair](../REPRODUCE.md#direct-bottle-repair). Original checkpoint files are not rewritten.
