# Measured tabletop transfer

Call `decide(o)` once per decision and preserve its returned `state`. Start each new transfer or homing operation with fresh state only after reconciling any held object.

- Supply calibrated grasp-center source/destination targets and a signed visual upright correction. The default closing axis is perpendicular to the unsigned silhouette axis; this is a grasp heuristic, not verified contact geometry.
- The sequence opens, clears vertically, approaches, closes, lifts, rotates at clearance, transits, descends, releases, and retreats toward the active arm's measured home workspace.
- Fresh TCP/jaw measurements and receipts gate progress. Small tracking failures get bounded retries; reach/collision failures return for replanning without opening a held object.
- Jaw obstruction is only evidence of a grasp, not proof. RGB inspection must confirm the object moved, landed centrally, and is upright. TCP-only input cannot independently verify wrist yaw or final object orientation.
- Clearance is relative to the table, not a full obstacle model. Collision-aware TCP planning remains essential. Direct homing uses small measured joint increments and conservative proximity/height checks, which do not prove swept-volume safety.
- Crossover transfers may require an agent-planned, visually verified staging location and regrasp. No unseen handoff location is invented. Park an obstructing inactive arm through a separately observed recovery.
- `home` requires initially open, known-empty grippers; it verifies both measured joint vectors against `home_q` and both jaws open before finishing.

`done` means the commanded sequence completed, not judge success. This initial skill has not been robot-tested.
