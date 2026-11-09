# Replay the archived SmolVLA edits

Run the commands from `experiments/libero_goal/`. Use the [pinned runtime versions](../actor-provenance.json), set `LIBERO_SOURCE` to the pinned LIBERO-PRO checkout, and place the pinned checkpoint files under `src/models/smolvla_libero` and the VLM config/tokenizer under `src/models/smolvlm2`. The [experiment page](../../README.md#方法与复现) links the upstream revisions. The actor loads offline; choose a CUDA device and a compatible EGL renderer. No new editor request or model download occurs during replay.

## Direct bottle repair

```sh
python src/replay_locked.py weight-repair --out runs/direct --render-gpu 0
```

This reruns the diagnostic reference and both numerical proposals, then checks the conservative version for 40 extra steps. The [original match inputs](replay-weight-screen/) contain the stock traces and initial observations used to validate the matched prefix. The focal bottle edit is deliberately installed after 80 controls, unlike the cold-start evaluations below.

## Fixed bottle/bowl composition

```sh
python src/replay_locked.py multitask-repair --out runs/multitask --render-gpu 0
```

The four-arm score requires matching initial observations within each case. Two separate 40-step unheld continuations are reported under `runs/multitask/retention/`; they explain the corrected stable counts in the [published scorecard](multitask-repair/scorecard.json).

## Sequential repair

```sh
python src/replay_locked.py sequential-repair --out runs/sequential --render-gpu 0
```

The command reconstructs the v3 checkpoint from the upstream model plus the archived absolute values. It runs the original, bottle, bowl, and joint versions on the same case definitions.

## BP control and activation localization

```sh
python src/replay_locked.py bp-control --out runs/bp --render-gpu 0
python src/replay_locked.py activation-repair --out runs/localization --render-gpu 0
# Or replay just the last Astra direct-edit arm:
python src/replay_locked.py activation-repair --arm astra-wsp \
  --out runs/localization-direct --render-gpu 0
```

Full replays produce raw episode videos and a paired `scorecard.json`. The published cold-start and matched-grasp evaluations are separate; the latter is run by `src/bp-control/postgrasp.py` with copies of the [teacher manifest](bp-control/teacher-evidence/teachers.json) and the locked `bp.json` and `wsp.json` in one writable directory.

### Rebuild the shared inputs

```sh
e=results/repair-series
mkdir -p runs/bp-inputs
python src/bp-control/bp_fit.py prepare \
  --teachers "$e/bp-control/teacher-evidence/teachers.json" \
  --coords "$e/sequential-repair/v3.json" --stock-dir "$e/replay-stock" \
  --dataset runs/bp-inputs/training.pt --shared runs/bp-inputs/shared.json --render-gpu 0
python src/bp-control/bp_fit.py fit \
  --dataset runs/bp-inputs/training.pt --out runs/bp-inputs/bp.json
python src/activation-repair/probe_activations.py \
  --out runs/activation-inputs --dataset runs/bp-inputs/training.pt --render-gpu 0
python src/activation-repair/fit.py select-auto \
  --localization runs/activation-inputs/localization.json --out runs/activation-inputs/auto-selection.json
python src/activation-repair/fit.py fit \
  --localization runs/activation-inputs/localization.json --dataset runs/bp-inputs/training.pt \
  --selection runs/activation-inputs/auto-selection.json --out runs/activation-inputs/auto-bp.json
```

Use the archived `astra-selection.json` in the last command to reproduce Astra-position BP. To measure action leakage on the eight extra fixed observations, use `src/activation-repair/combined_protection.py --run DIR --render-gpu 0` with the three locked edits and `localization.json` copied to a writable `DIR`.

### Optional new editor requests

The archived coefficients are the ones actually evaluated. To generate fresh proposals, install the compatible `openai_codex` SDK in an authenticated Codex environment. Set `CODEX_BIN` if the executable is not on PATH and `CODEX_SDK_DIR` only if the SDK is not installed in that Python environment. The `ask_*.py` entry points accept archived probe/target files; each call uses GPT-6 Astra and writes a new local audit. A new proposal is a new trial, not the archived result.
