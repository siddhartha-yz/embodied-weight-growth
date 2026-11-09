# Recover without dropping

Preserve `state` between decisions. This revision accepts fresh state or the previous initialization-only “no history” refusal.

- An obstructed jaw with no history is treated as a possible held object. Do not open or descend to the supplied projected source.
- Stationary public pose previews search for a top-down yaw whose candidate absolute joints closely match measured joints. Coarse/fine queries are motionless; ambiguous or poor matches stop safely. This is an approximate orientation estimate, not verified forward kinematics.
- Recovery keeps the grip closed, clears vertically, and carries to destination clearance. It then requests visual center/contact and upright verification. The supplied uncertain zero turn does **not** authorize release.
- To continue after that inspection, preserve the returned yaw/history. The agent must explicitly reconcile the new signed correction and clear the halt before resuming at `rotation_check` with `goal = yaw + correction`; do not restart a tabletop pickup.
- Normal transfers retain deliberate approach, close, lift, incremental rotation, placement, release, and own-workspace retreat. Failed paths are not blindly repeated; tracking/timeouts receive bounded retries based on fresh measurements.
- Homing requires empty/open jaws and verifies both measured `home_q` vectors and open jaws. Direct joint increments have conservative height/proximity guards, not a swept-volume guarantee.

Preview matching and recovery have not been robot-tested. Final centering and upright orientation still require RGB acceptance.
