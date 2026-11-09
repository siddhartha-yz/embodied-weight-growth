# Identity-safe tabletop transfer

**Required next action:** suspend the requested 7 transfer. Visually reconcile the held 8, obtain its calibrated destination and upright correction, and explicitly plan its recovery while retaining the grip. Do not release it, relabel it, or send it to 7's destination.

- Preserve state across decisions. A halted state requires agent reconciliation, not repeated retries or blind state reset.
- Fresh transfers require an open active gripper. Proprio and jaw width cannot establish held-object identity.
- Acquisition history is recorded only after this skill's approach and nonempty close; it remains conditional on correct visual source identification.
- Normal execution: clear, approach, close, lift, carry closed, rotate incrementally at clearance, place, release, retreat. Preview candidates never authorize direct joint bypasses.
- Tracking/timeouts use fresh target error and bounded retries. Infeasible paths return for revised geometry without opening the grip.
- Homing requires empty/open jaws and verifies both measured home joint vectors. Height/proximity checks are conservative heuristics, not swept-volume guarantees.

The present transfer input lacks the held object's recovery target and reliable orientation. Therefore `fail_reobserve` is intentional; no safe automatic recovery is claimed. Verify identity, centering, and upright orientation visually before advancing the queue.
