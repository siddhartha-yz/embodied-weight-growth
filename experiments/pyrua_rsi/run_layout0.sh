#!/usr/bin/env bash
# Run the released PyRUA-RSI task program through the original native eval adapter.
set -euo pipefail
umask 077

run_id=${1:?usage: run_layout0.sh NEW_RUN_ID IDLE_GPU}
gpu=${2:?supply an already verified idle physical GPU}
[[ "$run_id" =~ ^pyrua-rsi-[A-Za-z0-9_-]+$ && "$gpu" =~ ^[0-9]+$ ]]
work=${ROBO_WORK_ROOT:-"$HOME/data/robodojo-work"}
pin=a8ed9fab9084c971d684136faf6d21e1afabd5b0
pyrua="$work/src/PyRUA-Lean"
sim="$work/src/RoboDojo"
run="$work/archive/pyrua/$run_id"
python="$work/miniforge3/envs/PyRUALean/bin/python"
task=arrange_largest_number

test -x "$python"
test -f "$sim/Assets/Eval_Layout/RoboDojo/arx_x5/0/${task}_0.json"
export SAM3_CHECKPOINT_PATH="$work/downloads/pyrua/sam3.pt"
test -s "$SAM3_CHECKPOINT_PATH"
mkdir -- "$run"  # never overwrite an attempt
mkdir -- "$run/source" "$run/tmp"

git -C "$pyrua" archive "$pin" src/pyrualean integrations/xpolicylab/PyRUA_RSI |
  tar -xf - -C "$run/source"
export PYTHONPATH="$run/source/src"
export PATH="$work/miniforge3/bin:$work/downloads/ffmpeg-static:$PATH"
export TMPDIR="$run/tmp" PYTHONDONTWRITEBYTECODE=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=N
export EVAL_NUM=1 EVAL_ENV_TYPE=sim ROBODOJO_RENDER_GPU="$gpu" ROBODOJO_MAX_BASH_RETRIES=1
export ROBODOJO_RUN_ID="$run_id" PYRUALEAN_RESULT_DIR="$run/policy-records"
# Existing assets only. XPolicyLab and the released offline responder use loopback.
export http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 all_proxy=http://127.0.0.1:9
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy"
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
unset OPENAI_API_KEY OPENAI_BASE_URL PYRUALEAN_API_KEY_ENV PYRUALEAN_BASE_URL

"$python" - "$run" "$sim" "$pin" "$gpu" "$task" <<'PY'
import json
from pathlib import Path
import subprocess
import sys
from pyrualean.hosts.robodojo_client_root import lay_out

run, simulator = map(Path, sys.argv[1:3])
pin, gpu, task = sys.argv[3:]
policy = run / "source/integrations/xpolicylab/PyRUA_RSI"
layout = simulator / "Assets/Eval_Layout/RoboDojo/arx_x5/0"
available = list(layout.glob(f"{task}_*.json"))
if not available or not (layout / f"{task}_0.json").is_file():
    raise RuntimeError("standard layout 0 not available")
def revision(repo):
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
(run / "protocol.json").write_text(json.dumps({
    "task": task, "eval_seed": 0, "layout": 0, "episodes": 1, "gpu": int(gpu),
    "policy": "PyRUA_RSI", "policy_revision": pin, "program": f"programs/{task}.py",
    "model": "offline-program (loopback replay; no LLM)", "observation": "default RGB and proprioception",
    "native_limit": 1050, "simulator_revision": revision(simulator),
    "xpolicylab_revision": revision(simulator / "XPolicyLab"), "remote_run": str(run),
}, indent=2) + "\n")
lay_out(run / "bench", simulator, depth=False, policy_dir=policy)
PY

printf 'PyRUA-RSI run: %s\n' "$run"
cd "$run/bench"
status=0
bash XPolicyLab/policy/PyRUA_RSI/eval.sh \
  RoboDojo "$task" offline arx_x5 joint 0 "$gpu" "$gpu" PyRUALean RoboDojo \
  >"$run/eval.log" 2>&1 || status=$?
printf '%s\n' "$status" >"$run/launcher.exit"
tail -35 "$run/eval.log"
exit "$status"
