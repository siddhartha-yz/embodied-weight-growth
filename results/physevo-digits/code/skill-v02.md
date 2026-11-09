# Feedback-aware transfer recovery

- Preserve returned state on every call. For this recovery, pass the previous failed transfer state: migration retains the confirmed grasp yaw and original upright correction. Do not reset state or open the jaw.
- The new source estimate describes an occluded, held object—not a verified tabletop pickup. Migration ignores it. The supplied zero turn was uncertain, so it does not override the original correction.
- Transfer: approach from above, close, lift, carry without changing yaw, rotate incrementally at destination clearance, place, release, and retreat toward measured home workspace.
- Rotation previews compare equivalent **2π** yaw representations against measured absolute joints. They select a planner request, not a direct joint bypass. A preview is not proof of collision-free motion.
- Tracking/timeouts use measured TCP error and bounded retries. Position alone never confirms rotation. Infeasible paths return for new geometry without opening the held object.
- With fresh state and a closed jaw, request prior history rather than inventing current wrist yaw. Cross-arm handoffs still require observed staging geometry.
- Homing parks each empty arm toward its own workspace, then uses small measured joint increments to reach actual `home_q`; both jaws must finish open. Proximity checks do not certify swept-volume safety.

Jaw contact is not proof of retention, and completion is not visual success. Reobserve object centering and upright orientation after placement. No new robot execution was performed for this revision.
