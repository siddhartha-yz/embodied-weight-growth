# Preserve held-object identity

**Current episode:** stop the transfer queue. Wrist RGB shows 8 held; 7 remains on the table. Resolve the held object's own destination and upright correction before attempting the next pickup. Neither its destination nor orientation can be recovered from the current 7-transfer input.

- Preserve returned state across decisions. Never reset a failed held transfer merely to advance the queue.
- Fresh transfers require an open active gripper. Jaw obstruction indicates possible occupancy—not which digit is held. This skill cannot identify objects from proprio alone.
- Legacy recovery states are not trusted: their requested-digit labels were assigned without acquisition evidence. Do not relabel them or clear the halt blindly.
- Normal transfers record acquisition only after an observed approach and completed nonempty close. They lift, carry closed, preview incremental rotation at clearance, place, release, and retreat toward measured home workspace.
- Tracking/timeouts compare measured TCP against the target and permit bounded retries. Infeasible paths require revised geometry; no planner bypass or invented handoff.
- Homing requires empty/open grippers and verifies both actual home joint vectors and open jaws. Small joint increments and proximity guards do not certify swept-volume safety.

`done` confirms the commanded sequence, not visual correctness. Verify object identity, centering, and upright orientation before advancing.
