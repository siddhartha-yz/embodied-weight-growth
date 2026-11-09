# H1 numerical policy format

Send one complete JSON object per policy. It contains numbers and sparse connections; the host runs the same forward pass every 0.02 simulated seconds. No script, expressions, state updates, or optimization are run by this format.

```json
{
  "format": "h1-wsp-v1",
  "hidden": {"example": {"activation": "sin", "bias": 0, "weights": {"t": 1}}},
  "outputs": {"torso": {"bias": 0, "weights": {"example": 0.01}}}
}
```

This is syntax only, not a standing or walking policy. For each hidden unit, the host applies its activation (`tanh`, `sin`, or `relu`) to `bias + sum(weight × input)`. For each actuator, it clips `bias + sum(weight × feature)` to `[-1,1]`. Output features can be raw inputs or named hidden units. Hidden units only take raw inputs, so one feature can be reused by any number of outputs. `sin` takes radians.

**Inputs:** `qpos_0` through `qpos_25`, `qvel_0` through `qvel_24`, `delta_0` through `delta_25` (current qpos minus that episode's initial qpos), and `t` (elapsed simulated seconds since reset). No other features are computed. Outputs are keyed by the actuator `joint` names in the public `action.channels` table. The host sends normalized **position targets**, not torques. If the channel range is `[lo, hi]`, target angle `q` radians corresponds to normalized `2*(q-lo)/(hi-lo)-1`; a constant target goes in the output bias. Zero means the midpoint of that actuator's range, not its initial angle.

Omitted `hidden`, `bias`, or `weights` fields are empty or zero. An omitted output is exactly zero, not a hold command. Hidden names are 1–32 ASCII letters, digits, or underscores, start with a letter, and cannot shadow an input. Limits: 128 hidden units, 4,096 explicitly supplied coefficients and biases, numeric range `[-10000,10000]`, and 128 KiB of ASCII JSON (a submission tool may impose a smaller limit). Duplicate keys, unknown names/fields, and nonfinite values are rejected.
