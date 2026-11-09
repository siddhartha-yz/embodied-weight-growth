"""Verify matched episode starts and aggregate BP/WSP/stock results."""
import argparse
import json
from pathlib import Path
import numpy as np

ARMS=("stock","bp","wsp")

def load(path):
    return json.loads(path.read_text())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--stock-dev",type=Path,help="reuse stock development rollouts from this directory")
    p.add_argument("--stock-preserve",type=Path,help="reuse stock preservation rollouts from this directory")
    args=p.parse_args()
    run=args.run.resolve()
    roots={arm:{split:run/f"{arm}-{split}" for split in ("dev","heldout","preserve")} for arm in ARMS}
    roots["stock"]["dev"]=args.stock_dev or run/"stock-dev"
    roots["stock"]["preserve"]=args.stock_preserve or run/"stock-preserve"
    results={}
    for arm, splits in roots.items():
        results[arm]={}
        for split,folder in splits.items():
            for row in load(folder/"summary.json"):
                case=row["id"]
                assert case not in results[arm]
                group=("heldout_"+row["goal"] if split=="heldout" else "old"
                       if split=="preserve" or row["kind"]=="clean" else row["kind"])
                path=folder/case
                assert (path/"rollout.mp4").stat().st_size>0
                results[arm][case]=(row,path,group)
    cases={}; counts={}
    assert all(set(x)==set(results["stock"]) for x in results.values())
    for case,(ref,ref_dir,group) in results["stock"].items():
        scores={}
        with np.load(ref_dir/"initial-public.npz",allow_pickle=False) as public:
            for arm in ARMS:
                row,path,g=results[arm][case]
                assert group==g
                assert all(row.get(k)==ref.get(k) for k in
                    ("id","seed","goal","init_source","init_index","layout","instruction","first_prompt","prompt_mode"))
                with np.load(path/"initial-public.npz",allow_pickle=False) as obs:
                    assert set(obs.files)==set(public.files)
                    assert all(np.array_equal(obs[k],public[k]) for k in obs.files),(case,arm)
                fields=("native_success","stable_released_40","stable_released_at","first_native_goal",
                        "first_lift","first_lift_step","first_cabinet_contact_after_lift","last_contact",
                        "steps","final_native")
                scores[arm]={k:row.get(k) for k in fields}|{"rollout":str(path/"rollout.mp4")}
                a=counts.setdefault(group,{}).setdefault(arm,{"n":0,"native":0,"stable":0,"target_first_lift":0})
                a["n"]+=1
                a["native"]+=bool(row.get("native_success"))
                a["stable"]+=bool(row.get("stable_released_40"))
                a["target_first_lift"]+=bool(row.get("goal") and row.get("first_lift")==row["goal"])
        cases[case]={"group":group,"arms":scores}
    assert all(counts[g]["stock"]["n"]==n for g,n in {"history":1,"dev_bowl":1,"old":8}.items())
    assert all(counts[g]["stock"]["n"]>=3 for g in ("heldout_bottle","heldout_bowl"))
    out={"cases_n":len(cases),"rollouts_n":len(cases)*len(ARMS),"totals":counts,"cases":cases}
    (run/"scorecard.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(counts,indent=2))

if __name__=="__main__":
    main()
