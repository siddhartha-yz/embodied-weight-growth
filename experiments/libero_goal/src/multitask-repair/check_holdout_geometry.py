"""Validate initial support/contact only; do not run or inspect any policy actions."""
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from layouts import BowlOffsetEnv

cases=json.loads(Path(sys.argv[1]).read_text())
result=[]
with BowlOffsetEnv(pair="objects",render=False) as env:
    for c in cases:
        r={"id":c["id"],"valid":False}
        try:
            env.reset(goal=c["goal"],init_index=c["init_index"],seed=c["seed"],
                      init_source="bowl",layout=c["layout"])
            if env.status()["goal_achieved"]:
                raise ValueError("Already achieved target at reset")
            r["valid"]=True
        except (RuntimeError,ValueError) as ex:
            r["reason"]=str(ex)
        result.append(r)
        print(json.dumps(r),flush=True)
Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+"\n")
if not all(r["valid"] for r in result): sys.exit(2)
