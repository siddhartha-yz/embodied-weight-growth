"""One Codex SDK call to propose direct FP32 edits on the shared BP/WSP evidence."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
from time import perf_counter

HELPER=Path(__file__).resolve().parents[1]/"codex_transport.py"
spec=importlib.util.spec_from_file_location("established_astra_sdk",HELPER)
transport=importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
obj=transport.object_schema
transport.EFFORT="high"
transport.BASE=("You are the direct numerical weight editor in a controlled SmolVLA experiment. "
                "Use only supplied evidence. Return absolute weight values and a concise diagnosis; do not use tools, write code or train.")
transport.SCHEMA=obj({"edits":{"type":"array","items":obj({"row":{"type":"integer"},
        "column":{"type":"integer"},"value":{"type":"number"}})},"diagnosis":{"type":"string"}})
transport.parse_answer=json.loads
INSTRUCTIONS="""Repair post-grasp transport of both the wine bottle and the bowl, and preserve the
original grasp and ordinary placement behavior. Starting from the original SmolVLA, assign one
shared set of absolute values in existing FP32 model.action_out_proj.weight. Only the 16 coordinates
listed below are editable. Each value must be within 0.65 of its stock value; unlisted/omitted
coordinates stay stock. No other parameters, new modules, per-object routing, adaptive schedules,
runtime gates, tools, code or training. Every edit is in effect before step one.

Use the actual successful rescue action chunks found by short counterfactual simulations after a
correct grasp. Use the supplied exact same observation and noise action references. Retention
targets are the stock action chunks, especially during approach/grasp and normal placement.
The numbers are seven-dimensional native actions (x,y,z,rx,ry,rz,gripper), not Cartesian meters.
Positive gripper action closes. The rollout executes the first 10 controls then replans. A positive
scalar change need not produce a positive action change. Responses can couple axes.
Use the shared loss definitions and prefer balancing both objects and preservation instead of
only matching the largest discrepancy. If local probes are given, they are for the stock model
and do not guarantee additivity or closed-loop success. You have one proposal; blind layouts
are not provided. Return only numeric absolute values and a short diagnosis.
"""

def compact_evidence(shared):
    """Give exact targets plus the local quadratic implied by the supplied action probes."""
    probes=shared["probes"]
    keys=lambda rows:[(e["row"],e["column"]) for e in rows]
    assert keys(probes)==keys(shared["coordinates"])
    groups={}
    for x in shared["contexts"]:
        groups.setdefault(x["group"],[]).append(x)
    combined=[[0.0 for _ in probes] for _ in probes]
    detail={}
    group_weights=shared["objective"]["group_weights"]
    for g, contexts in groups.items():
        residual=[]; effects=[[] for _ in probes]
        for x in contexts:
            for t,(stock,target) in enumerate(zip(x["stock"],x["target"],strict=True)):
                for a in x["axes"]:
                    residual.append(stock[a]-target[a])
                    for i,q in enumerate(probes):
                        effects[i].append(q["effects"][x["id"]][t][a]/q["step"])
        def mean_dot(a,b):
            return sum(x*y for x,y in zip(a,b,strict=True))/len(a)
        linear=[2*mean_dot(v,residual) for v in effects]
        diagonal=[mean_dot(v,v) for v in effects]
        w=group_weights[g]
        detail[g]={"constant":round(mean_dot(residual,residual),7),
                   "linear": [round(x,7) for x in linear],
                   "quadratic_diagonal":[round(x,7) for x in diagonal]}
        for i,v in enumerate(effects):
            for j,u in enumerate(effects):
                combined[i][j]+=w*mean_dot(v,u)
    return {**{k:v for k,v in shared.items() if k!="probes"},
            "local_estimate_from_single_weight_plus_0.20_probes":{
                "notation":"offset d = absolute_value minus stock. In coordinate-list order, each group has "
                "L(d) ≈ constant + linear·d + dᵀQ_group d. Matrix below is the weighted sum of Q_group. "
                "Negative linear entry favors positive offset, and vice versa. Only the local approximation is linearized.",
                "groups":detail,"total_quadratic":[[round(x,7) for x in row] for row in combined]}}

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("shared",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    auditfile=a.out.with_suffix(".audit.json")
    if a.out.exists() or auditfile.exists():
        p.error("This comparison already has a WSP answer or SDK audit")
    shared=json.loads(a.shared.read_text())
    table={(e["row"],e["column"]):e["stock"] for e in shared["coordinates"]}
    assert len(table)==16
    prompt=INSTRUCTIONS+"\nShared training evidence (same targets as BP): "+json.dumps(compact_evidence(shared),separators=(",",":"))
    a.out.with_suffix(".prompt.txt").write_text(prompt)
    sdk=transport.import_sdk()
    audit={"model_requested":transport.MODEL,"effort":transport.EFFORT,"answer":None}
    start=perf_counter()
    try:
        answer=transport.ask(sdk,prompt,[],audit)
        audit["answer"]=answer
        seen=set()
        for e in answer["edits"]:
            k=(e["row"],e["column"]); v=e["value"]
            if k not in table or k in seen or type(v) not in (int,float) or not math.isfinite(v) or abs(v-table[k])>.650001:
                raise ValueError(f"Unlisted, duplicate or out-of-bound assignment {e}")
            e["value"]=struct.unpack("f",struct.pack("f",v))[0]
            seen.add(k)
        dump(a.out,answer)
        print(f"Astra assigned {len(seen)} of 16 stock positions",flush=True)
    except Exception as exc:
        audit["error"]=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        audit["wall_s"]=perf_counter()-start
        dump(auditfile,audit)

if __name__=="__main__":
    main()
