# Slow gait probe: unsuccessful

Reproduce with ordered includes skills/h1_stance.py, skills/h1_slow_probe.py. Initialize h1_slow_probe_init from reset observation, then call h1_slow_probe_action at 50 Hz. Stop on termination/truncation or unsafe posture. This is a failed experimental controller, not a recommended walking skill.

Attempt 11: at 4.4 seconds, displacement +0.15606 m, height 0.98255 m, pitch 0.08262 rad, vx 0.07358 m/s. At 5.2 seconds, displacement +0.24421 m, pitch 0.13619 rad, vx 0.20321 m/s. It then accelerated and lost height. Safety stop at 5.84 seconds: displacement +0.69612 m, height 0.78902 m, pitch approximately 0.355 rad, vx 1.46367 m/s. Native success remained false; no native fall termination had occurred at the safety stop. The saved function matched the exploratory implementation exactly (maximum action difference 0).

Conclusion: reducing periodic amplitude and reference speed delayed instability but did not solve walking. Forward displacement and momentary speed near 1 m/s were associated with falling, not controlled locomotion. Only the separate stationary controller was verified upright for the full 20-second episode. Remaining physical budget was 15 steps, insufficient for meaningful gait validation. No successful walking controller was found.
