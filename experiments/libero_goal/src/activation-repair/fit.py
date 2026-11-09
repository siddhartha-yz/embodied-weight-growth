"""Select and fit action-output feature pairs under the shared activation-pilot objective."""
from collections import Counter
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"bp-control"))
import bp_fit as base

from scipy.optimize import lsq_linear
import numpy as np
import torch

GROUP_WEIGHTS={"repair-bottle":.5,"repair-bowl":.5,"grasp-protect":5.,"normal-carry":1.}
ROWS=(1,2)
BOUND=.65
PAIR_CAP=8
STEPS=32


def group_for(context):
    if context["group"] in ("repair-bottle","repair-bowl"):
        return context["group"]
    if context["case"].startswith("clean-") and context["t"] >= 70:
        return "normal-carry"
    return "grasp-protect"


def load(path):
    return json.loads(path.read_text())


def save(path,obj):
    if path.exists():
        raise FileExistsError(path)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def keyed(row):
    return row["row"],row["column"]


def linear_system(evidence):
    coords=sorted(evidence["coordinates"],key=keyed)
    probes={keyed(p):p for p in evidence["probes"]}
    if {keyed(c) for c in coords}!=set(probes):
        raise ValueError("Every candidate must have the same signed intervention data")
    ctx=evidence["contexts"]
    counts=Counter(group_for(x) for x in ctx)
    features=[]; targets=[]
    for x in ctx:
        ax=x["axes"]; group=group_for(x)
        y=(np.asarray(x["target"])-np.asarray(x["stock"]))[:,ax].reshape(-1)
        scale=np.sqrt(GROUP_WEIGHTS[group]/(counts[group]*len(y)))
        targets.append(y*scale)
        features.append(np.stack([np.asarray(probes[keyed(c)]["effects"][x["id"]])[:,ax].reshape(-1)
                                   /probes[keyed(c)]["step"] for c in coords],axis=1)*scale)
    return coords,np.concatenate(features,axis=0),np.concatenate(targets)


def auto_select(evidence):
    """Greedy paired pursuit with a bounded least-squares refit after each added pair."""
    coords,A,y=linear_system(evidence)
    by_col={c:[i for i,e in enumerate(coords) if e["column"]==c] for c in sorted({e["column"] for e in coords})}
    if len(by_col)<PAIR_CAP or any(tuple(coords[i]["row"] for i in inds)!=ROWS for inds in by_col.values()):
        raise ValueError("Localization must supply at least eight eligible Y/Z feature pairs")
    remaining=list(by_col); cols=[]; picked=[]; history=[]
    for _ in range(PAIR_CAP):
        trials=[]
        for c in remaining:
            k=picked+by_col[c]
            opt=lsq_linear(A[:,k],y,bounds=(-BOUND,BOUND),tol=1e-10)
            if not opt.success:
                raise ValueError(f"Constrained selection did not converge for candidate {c}: {opt.message}")
            trials.append((float(2*opt.cost),c,k))
        loss,c,picked=min(trials,key=lambda r:(r[0],r[1]))
        remaining.remove(c); cols.append(c)
        history.append({"column":c,"linearized_objective":loss})
    return {"columns":cols,"selection_history":history,"objective_weights":GROUP_WEIGHTS,
            "stock_linearized_objective":float(y@y),"rule":"bounded paired forward selection"}


def require_selection(path):
    sel=load(path)
    cs=sel["columns"]
    if len(cs)!=PAIR_CAP or len(set(cs))!=PAIR_CAP or not all(type(c) is int for c in cs):
        raise ValueError(f"Expected exactly {PAIR_CAP} distinct integer columns")
    return cs


def run_fit(args):
    evidence=load(args.localization)
    cols=require_selection(args.selection)
    eligible={keyed(c):c for c in evidence["coordinates"]}
    keys=[(r,c) for r in ROWS for c in cols]
    if any(k not in eligible for k in keys):
        raise ValueError("A selected Y/Z pair is not in the measured candidate pool")
    dataset=torch.load(args.dataset,weights_only=True)
    rows=dataset["contexts"]
    refs={x["id"]:x for x in evidence["contexts"]}
    # No extra contexts introduced by either method: compare exact supervised quantities.
    if {x["id"] for x in rows}!=set(refs):
        raise ValueError("Cached training data and localization context IDs differ")
    for x in rows:
        ref=refs[x["id"]]
        if any(not np.array_equal(x[k],ref[k]) for k in ("stock","target","axes")) or group_for(x)!=group_for(ref):
            raise ValueError(f"Mismatched supervised context {x['id']}")
        x["group"]=group_for(x)
    base.GROUP_WEIGHTS=GROUP_WEIGHTS
    actor=base.SmolActor(n_action_steps=10)
    layer=base.head(actor)
    coordinates=[{"row":r,"column":c,"stock":float(layer.weight[r,c])} for r,c in sorted(keys)]
    if any(q["stock"]!=eligible[keyed(q)]["stock"] for q in coordinates):
        raise ValueError("Candidate stock weight mismatch")
    if args.out.exists():
        raise FileExistsError(args.out)
    # Reuse exact single-context training loop with the new immutable group and coordinate definitions.
    slim=args.out.with_suffix(".fit-data.pt")
    torch.save({"coordinates":coordinates,"contexts":rows},slim)
    args.dataset=slim
    args.batch=1
    args.steps=STEPS
    try:
        base.fit(args,actor)
    finally:
        slim.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    a=sub.add_parser("select-auto")
    a.add_argument("--localization",type=Path,required=True)
    a.add_argument("--out",type=Path,required=True)
    f=sub.add_parser("fit")
    f.add_argument("--localization",type=Path,required=True)
    f.add_argument("--dataset",type=Path,required=True)
    f.add_argument("--selection",type=Path,required=True)
    f.add_argument("--out",type=Path,required=True)
    f.add_argument("--score-edits",type=Path)
    args=p.parse_args()
    if args.mode=="select-auto":
        result=auto_select(load(args.localization))
        save(args.out,result)
        print(json.dumps(result),flush=True)
    else:
        run_fit(args)

if __name__=="__main__":
    main()
