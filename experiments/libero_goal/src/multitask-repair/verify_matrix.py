"""Join locked rollout partitions; assert each paired comparison has identical actual inputs."""
import argparse
import json
from pathlib import Path
import numpy as np

ARMS=("stock","bottle","bowl","merged")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("partitions",nargs="+",help="run-dir:included-arm,arm")
    args=p.parse_args()
    by={}
    for source in args.partitions:
        root,allowed=source.rsplit(":",1)
        folder=Path(root)
        for row in json.loads((folder/"summary.json").read_text()):
            arm=row["arm"]
            if arm not in allowed.split(","): continue
            case=by.setdefault(row["id"],{})
            if arm in case: raise ValueError(f"Duplicate {row['id']}/{arm}")
            row["rollout"]=(folder/arm/row["id"]/"rollout.mp4").as_posix()
            case[arm]=row
    slim={}; totals={}
    for cid,rows in by.items():
        if set(rows)!=set(ARMS): raise ValueError(f"Missing arms for {cid}: {set(ARMS)-set(rows)}")
        fields=("id","kind","init_index","seed","instruction","layout","first_prompt","prompt_mode","suite","task")
        s={k:rows["stock"].get(k) for k in fields}
        path=Path(rows["stock"]["rollout"]).with_name("initial-public.npz")
        with np.load(path,allow_pickle=False) as ref:
            for arm in ARMS:
                row=rows[arm]; movie=Path(row["rollout"])
                assert movie.is_file() and movie.stat().st_size>0
                assert all(row.get(k)==s[k] for k in fields), (cid,arm)
                with np.load(movie.with_name("initial-public.npz"),allow_pickle=False) as other:
                    assert ref.files==other.files and all(np.array_equal(ref[k],other[k]) for k in ref.files), (cid,arm)
        group=f"heldout_{rows['stock']['goal']}" if s["kind"]=="heldout" else "preservation" if s["kind"] in ("clean","clean_other") else s["kind"]
        slim[cid]={"paired_inputs_bit_identical":True,**s,"arms":{}}
        for arm,row in rows.items():
            by_group=totals.setdefault(group,{})
            t=by_group.setdefault(arm,{"n":0,"native":0,"correct_first_lift":0,"stable40":0,"stable_evaluable_n":0,
                                           "rescued_native_vs_stock":0,"lost_native_vs_stock":0})
            t["n"]+=1; t["native"]+=row["native_success"]
            t["correct_first_lift"]+=row.get("goal") is not None and row.get("first_lift")==row["goal"]
            t["stable40"]+=bool(row.get("stable_released_40"))
            t["stable_evaluable_n"]+="stable_released_40" in row
            t["rescued_native_vs_stock"]+=row["native_success"] and not rows["stock"]["native_success"]
            t["lost_native_vs_stock"]+=not row["native_success"] and rows["stock"]["native_success"]
            keys=("native_success","first_native_goal","stable_released_40","stable_released_at","first_lift","first_lift_step",
                "first_cabinet_contact_after_lift","last_contact","last_contact_xyz","first_open_after_lift","first_other_goal","final_native","final_xyz","steps","rollout")
            slim[cid]["arms"][arm]={k:row[k] for k in keys if k in row}
    payload={"totals":totals,"cases":slim}
    args.out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(totals,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
