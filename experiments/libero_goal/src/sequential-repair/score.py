"""Pair-check and aggregate the fixed-weight sequential comparison on the lab."""
import argparse
import json
from pathlib import Path
import numpy as np

VERSIONS=("v0","v1","v2","v3")

def load(path):
    return json.loads(path.read_text())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    a=p.parse_args()
    all_rows={}
    for v in VERSIONS:
        all_rows[v]={}
        for split,folder in (("dev",v),("preserve",v+"-preserve"),("heldout",v+"-heldout")):
            for row in load(a.run/folder/"summary.json"):
                case=row["id"]
                if case in all_rows[v]: raise ValueError(f"duplicate {v}/{case}")
                movie=a.run/folder/case/"rollout.mp4"
                assert movie.stat().st_size>0
                group=("old" if split=="preserve" or row["kind"]=="clean" else
                       "new_"+row["goal"] if split=="heldout" else row["kind"])
                all_rows[v][case]=(group,movie.parent,row)
    cases={}; counts={}
    for case,(group,ref,row0) in all_rows["v0"].items():
        scores={}
        with np.load(ref/"initial-public.npz",allow_pickle=False) as ref_np:
            for v in VERSIONS:
                g,folder,row=all_rows[v][case]
                assert g==group
                assert all(row.get(k)==row0.get(k) for k in
                    ("init_index","seed","init_source","layout","prompt_mode","instruction","first_prompt"))
                with np.load(folder/"initial-public.npz",allow_pickle=False) as test_np:
                    assert set(ref_np.files)==set(test_np.files)
                    assert all(np.array_equal(ref_np[k],test_np[k]) for k in ref_np.files),(case,v)
                scores[v]={k:row.get(k) for k in ("first_lift","first_lift_step",
                    "first_cabinet_contact_after_lift","last_contact","first_native_goal","native_success",
                    "stable_released_at","stable_released_40","final_native","steps")}
                scores[v]["rollout"]=str(folder.relative_to(a.run)/"rollout.mp4")
                acc=counts.setdefault(group,{}).setdefault(v,{"n":0,"native":0,"stable":0,"target_lift":0})
                acc["n"]+=1
                acc["native"]+=bool(row.get("native_success"))
                acc["stable"]+=bool(row.get("stable_released_40"))
                acc["target_lift"]+=bool(row.get("goal") and row.get("first_lift")==row["goal"])
        cases[case]={"group":group,"versions":scores}
    assert len(cases)==16 and all(set(rows)==set(cases) for rows in all_rows.values())
    assert {g:c["v0"]["n"] for g,c in counts.items()}=={
        "history":1,"dev_bowl":1,"old":8,"new_bowl":3,"new_bottle":3}
    payload={"paired_case_count":len(cases),"rollout_count":len(cases)*len(VERSIONS),
             "totals":counts,"cases":cases}
    (a.run/"scorecard.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(counts,indent=2))

if __name__=="__main__":
    main()
