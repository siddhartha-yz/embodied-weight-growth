"""Two Codex SDK calls: choose intervention-supported columns, then directly assign their values."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
from time import perf_counter

spec=importlib.util.spec_from_file_location("previous_sdk_objective",Path(__file__).resolve().parents[1]/"bp-control"/"ask_astra.py")
bp=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)
transport=bp.transport
obj=transport.object_schema
LOCATE=obj({"columns":{"type":"array","items":{"type":"integer"}},"diagnosis":{"type":"string"}})
ASSIGN=obj({"edits":{"type":"array","items":obj({"row":{"type":"integer"},
          "column":{"type":"integer"},"value":{"type":"number"}})},"diagnosis":{"type":"string"}})
W={"repair-bottle":.5,"repair-bowl":.5,"grasp-protect":5.,"normal-carry":1.}
COMMON="""Starting from stock SmolVLA, the task is to repair transport for wine bottle and bowl
without changing acquisition and normal transport. Existing FP32 action_out_proj Y/Z entries
are the only editable parameters. Deployment uses one fixed checkpoint from control 1, with
no activation hooks, external grasp detector, phase switch, new module, or per-task parameter
route. Interventions below were experimental probes only. The two tasks share the weights.
Select for effects on both repair targets and preservation, not simply the largest movement.
Absolute scalar changes are bounded to 0.65 from stock. Do not use tools, write code, or train.
"""
LOCATE_TEXT="""Localization stage: choose exactly eight distinct eligible columns, each authorizing
the existing Y and Z coefficients, for 16 scalars total. Choose where to edit, not the weights.
Use observed activation-contribution interventions and consider coverage across both repair tasks.
A feature correlated with carry is insufficient if intervening also changes grasp. The
quantitative evidence for selection is the same set of +0.20 development interventions available
to the automatic selector. Report which effects motivated the locations. Do not propose values.
"""
ASSIGN_TEXT="""Direct weight stage: you may assign absolute values only within the exact eight columns
already selected in the preceding stage. Unlisted values stay stock. You have one numeric
proposal. Action axes are native x,y,z,rx,ry,rz,gripper; positive gripper closes. Each context
gives the first 10 executed controls, seeded with the same original noise. Repair targets are
verified successful action branches, and preservation targets are stock actions. Grasp protection
is weighted 5x per group. The shared objective uses group means, as indicated. Approximate
quadratic coefficients summarize actual forward probes and are not a guarantee of closed-loop
effects. Aim for the two repair targets while keeping grasp and ordinary carry close to stock.
"""

def reweight(x):
    q={k:x[k] for k in ("coordinates","probes","objective","experiment","schema",
                         "batch_control_max_native_difference_from_single") if k in x}
    q["contexts"]=[]
    for r in x["contexts"]:
        g=r["group"]
        if g=="preserve":
            g="normal-carry" if r["case"].startswith("clean-") and r["t"]>=70 else "grasp-protect"
        if g in W: q["contexts"].append({**r,"group":g})
    assert set(r["group"] for r in q["contexts"])==set(W)
    q["objective"]={**x.get("objective",{}),"formula":"0.5*MSE(repair-bottle Y/Z)+0.5*MSE(repair-bowl Y/Z)+5*MSE(grasp-protect 7D)+1*MSE(normal-carry 7D)","group_weights":W}
    return q

def phase_summary(full,cols):
    rows=[]
    for q in full["signed_interventions"]:
        if q["column"] not in cols: continue
        row={k:q[k] for k in ("row","column","screen")}
        row["plus_0.20"]={k:{a:round(v,6) for a,v in x.items()
                             if a in ("rms_7d","teacher_relative_mse_drop")}
                           for k,x in q["signed"]["1"]["train"].items()}
        rows.append(row)
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("locate","write"))
    p.add_argument("evidence",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--selection",type=Path)
    a=p.parse_args()
    auditfile=a.out.with_suffix(".audit.json")
    if a.out.exists() or auditfile.exists():
        p.error("This comparison already has an SDK answer or audit at that path")
    is_loc=a.mode=="locate"
    if not is_loc and a.selection is None: p.error("Write requires the frozen localization")
    full=json.loads(a.evidence.read_text())
    data=reweight(full)
    cols={x["column"] for x in data["coordinates"]}
    if not is_loc:
        sel=json.loads(a.selection.read_text())
        cols=set(sel["columns"])
        data["coordinates"]=[x for x in data["coordinates"] if x["column"] in cols]
        data["probes"]=[x for x in data["probes"] if x["column"] in cols]
    key=lambda x:(x["row"],x["column"])
    data["coordinates"]=sorted(data["coordinates"],key=key)
    data["probes"]=sorted(data["probes"],key=key)
    allowed={key(x):x["stock"] for x in data["coordinates"]}
    assert len(allowed)==2*len(cols) and set(allowed)=={(r,c) for r in (1,2) for c in cols}
    packed=bp.compact_evidence(data)
    packed["phase_summary_same_training_pulses"]=phase_summary(full,cols)
    if is_loc:
        transport.SCHEMA=LOCATE
        text=LOCATE_TEXT
    else:
        transport.SCHEMA=ASSIGN
        packed["frozen_localization"]={"columns":sel["columns"],"diagnosis":sel["diagnosis"]}
        text=ASSIGN_TEXT
    transport.EFFORT="high"
    transport.BASE="You are the direct numerical weight editor in a controlled SmolVLA experiment. Use only the supplied experimental evidence and return the structured response. Do not run tools."
    prompt=COMMON+text+"\nExperimental evidence: "+json.dumps(packed,separators=(",",":"))
    a.out.with_suffix(".prompt.txt").write_text(prompt)
    sdk=transport.import_sdk()
    audit={"mode":a.mode,"model_requested":transport.MODEL,"effort":transport.EFFORT,"answer":None}
    start=perf_counter()
    try:
        response=transport.ask(sdk,prompt,[],audit)
        audit["answer"]=response
        if is_loc:
            chosen=response["columns"]
            if len(chosen)!=8 or len(set(chosen))!=8 or not set(chosen)<=cols:
                raise ValueError(f"Invalid location selection: {chosen}")
        else:
            seen=set()
            for e in response["edits"]:
                k=key(e); v=e["value"]
                if k not in allowed or k in seen or type(v) not in (float,int) or not math.isfinite(v) or abs(v-allowed[k])>.650001:
                    raise ValueError(f"Invalid numeric assignment: {e}")
                e["value"]=struct.unpack("f",struct.pack("f",v))[0]
                seen.add(k)
        bp.dump(a.out,response)
        print(a.mode,response.get("columns",f"{len(response.get('edits',[]))} assigned"),flush=True)
    except Exception as exc:
        audit["error"]=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        audit["wall_s"]=perf_counter()-start
        bp.dump(auditfile,audit)

if __name__=="__main__":
    main()
