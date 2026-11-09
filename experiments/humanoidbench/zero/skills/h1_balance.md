# H1 Playground evidence

Native walking success has NOT been achieved. skills/h1_stance.py is a stationary balance component, not a walking solution.

## Reproduction
Include skills/h1_stance.py. Immediately after reset, set c=h1_stance_init(obs). At each 0.02 s step use h1_stance_action(c,obs), update obs, and stop on termination/truncation. Initialize only from public observations; no absolute scene coordinates are stored.

## Verified stationary result
Attempt 10 survived all 1000 steps / 20 s without falling. Final pelvis height 0.96525 m, displacement -0.01208 m, pitch -0.04017 rad, roll -0.00052 rad, forward velocity 0.000095 m/s. Native success remained false. The function implementation was numerically equivalent to the exploratory controller (max joint-target difference 0). Hip roll drifted outward and height slowly decreased, so stability beyond 20 s is untested.

## Failures and constraints
- Holding measured initial joint targets drifted backward. Targets are position commands, not torques; normalized zero is not the initial posture.
- Ankle-only pitch feedback, both tested signs, did not stabilize the initial crouch. Hip pitch feedback plus root-position and velocity ankle correction was necessary for the verified stance.
- Hip yaw target +0.4*yaw+0.08*yaw_rate behaved much better than the negative sign, which excited rapid yaw oscillations.
- Larger open-loop alternating hip/knee sinusoids fell within about 2 seconds.
- A small gait (0.6 s period, hip cosine amplitude 0.09 rad, positive-sine knee lift 0.16 rad) with reference advancing 0.12 m/s moved +0.2745 m by episode second 4, but accelerated forward. Subsequent common hip foot-placement correction did not recover it; fall at 4.86 s. Do not mistake falling displacement for successful walking.
- Classes fail with NameError: __build_class__ not found. Use functions and dictionaries; imports of math and numpy worked.
