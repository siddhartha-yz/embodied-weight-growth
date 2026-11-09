"""Candidate feedback and fixed post-selection evaluation for weight surgery round 2."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np

import pilot as previous
from walk_policy import JOINT_ORDER, compile_weights

previous.MAX_EDITS = 24
previous.MAX_DELTA = .60
NOMINAL, TARGET = previous.NOMINAL, previous.TARGET
DEV_HEALTHY = [(271, .02), (272, .02)]
DEV_TARGET = [(271, .08), (302, .08)]
HELDOUT = [(s, p) for s, p in zip(range(320, 332), [.08]*6+[.07]*3+[.09]*3)]
RETENTION = [(s, p) for s, p in zip(range(320, 332), [.02]*6+[.015]*3+[.025]*3)]
RMS_LIMIT, PEAK_LIMIT, RATIO_MIN, RISK_MIN = .010, .060, 3., .015


def risky(s):
    # Identified from the old development trace; this is a measurement, not a deployed gate.
    return (s["vx"] > 1.45 and s["contact"] == [0, 1]) or (
        s["vx"] < -.12 and -.75 < s["pitch"] < -.30)


def torque_response(base, patch, trajectories, mask=lambda _: True):
    rows = [(s,np.asarray(patch(s,.50))-np.asarray(base(s,.50)))
            for r in trajectories for s in r["trace"] if mask(s)]
    if not rows:
        return {"n":0,"rms":None,"peak":None}
    u = np.asarray([z for _,z in rows])
    i,j = np.unravel_index(abs(u).argmax(), u.shape)
    return {"n":len(rows),"rms":float(np.sqrt(np.mean(u*u))),"peak":float(abs(u[i,j])),
            "peak_t":round(rows[i][0]["t"],4),"peak_joint":JOINT_ORDER[j]}


def summarize_case(case, before, after):
    return {"seed":case[0],"pitch":case[1],"before":previous.stats(before),"after":previous.stats(after)}


def dev(spec, audit, round_number, out):
    base = compile_weights(spec)
    cases = DEV_HEALTHY+DEV_TARGET
    originals = {c:previous.roll(spec,c) for c in cases}
    assert previous.stats(originals[NOMINAL])["pass"] and not previous.stats(originals[TARGET])["pass"]
    healthy = [originals[c] for c in DEV_HEALTHY if previous.stats(originals[c])["pass"]]
    failed = [originals[c] for c in DEV_TARGET if not previous.stats(originals[c])["pass"]]
    result = []
    for i, cand in enumerate(audit["answer"]["candidates"],1):
        name = f"r{round_number}-c{i}"
        row = {"id":name,"label":cand["label"],"diagnosis":cand["diagnosis"]}
        try:
            patched, change = previous.patch(spec,cand)
        except ValueError as e:
            result.append({**row,"rejected":str(e)})
            continue
        patch = compile_weights(patched)
        h = torque_response(base,patch,healthy)
        r = torque_response(base,patch,failed,risky)
        other = torque_response(base,patch,failed,lambda s: not risky(s) and abs(s["pitch"])<.75)
        ratio = (r["rms"] or 0.)/max(h["rms"],1e-12)
        h_results = [summarize_case(c,originals[c],previous.roll(patched,c)) for c in DEV_HEALTHY]
        d_results = []
        trace_snips = []
        for c in DEV_TARGET:
            r0,r1 = originals[c],previous.roll(patched,c)
            d_results.append(summarize_case(c,r0,r1))
            if c == TARGET:
                trace_snips = previous.state_summary(r1, [0,20,40,45,50,55,60,64,68,72,78,83,88,92,96,101,110,125,150,200,249])
        nom = h_results[0]
        keep = nom["after"]["pass"] and .85 <= nom["after"]["distance_m"]/nom["before"]["distance_m"] <= 1.15 and nom["after"]["alternating_steps"] >= nom["before"]["alternating_steps"]-2
        keep &= all(x["after"]["pass"] for x in h_results if x["before"]["pass"])
        quiet = h["rms"] <= RMS_LIMIT and h["peak"] <= PEAK_LIMIT
        selective = (r["rms"] or 0.) >= RISK_MIN and ratio >= RATIO_MIN
        target_pass = d_results[0]["after"]["pass"]
        previous.write(out/f"{name}.json",patched)
        result.append({**row,"edits":change,"healthy":h,"risk":r,"nonrisk_failure_trajectory":other,
                       "risk_healthy_ratio":ratio,"quiet":quiet,"retains":keep,
                       "selective":selective,"target_pass":target_pass,"qualified":quiet and keep and selective and target_pass,
                       "retention_dev":h_results,"target_dev":d_results,"target_trace_samples":trace_snips})
    previous.write(out/f"dev-{round_number}.json",result)
    for x in result:
        if "rejected" in x: print(x["id"],x["rejected"])
        else:
            t = x["target_dev"][0]["after"]
            print(x["id"],"keeps",x["retains"],"quiet",x["quiet"],"selective",x["selective"],
                  "RMS n/r",round(x["healthy"]["rms"],5),round(x["risk"]["rms"],5),
                  "target pass/s/m",t["pass"],round(t["elapsed_s"],2),round(t["distance_m"],2))


def lock(out):
    all_rows = []
    for f in sorted(out.glob("dev-*.json")):
        all_rows.extend(previous.read(f))
    valid = [r for r in all_rows if "rejected" not in r]
    if not valid:
        raise ValueError("no valid patch to evaluate")
    # Never choose by hidden cases: qualify on development, prefer preserving the gait,
    # then targeted state selectivity and measured rescue. Weak candidates are diagnostics only.
    def rank(r):
        target = r["target_dev"][0]["after"]
        return (r["qualified"],r["retains"] and r["quiet"],r["target_pass"],r["selective"],
                sum(c["after"]["pass"] for c in r["target_dev"]),target["elapsed_s"],
                target["alternating_steps"],target["distance_m"], -len(r["edits"]))
    top = max(valid,key=rank)
    previous.write(out/"selection.json", {"selection_data":"development only","candidate":top,"all":all_rows})
    print("Locked",top["id"],"qualified",top["qualified"])


def final(spec, out):
    pick = previous.read(out/"selection.json")["candidate"]
    patched = previous.read(out/f'{pick["id"]}.json')
    rows, traces = [], {}
    hidden_healthy, hidden_failing = [], []
    for group,cases in (("nominal",[NOMINAL]),("target",[TARGET]),("heldout",HELDOUT),("retention",RETENTION)):
        for c in cases:
            before,after=previous.roll(spec,c),previous.roll(patched,c)
            rows.append({"group":group,**summarize_case(c,before,after)})
            if group == "retention" and previous.stats(before)["pass"]:
                hidden_healthy.append(before)
            elif group == "heldout" and not previous.stats(before)["pass"]:
                hidden_failing.append(before)
            if group in ("target","nominal"):
                traces[group] = {"before":before,"after":after}
                previous.write(out/f"{group}-traces.json",traces[group])
    totals={}
    for group in ("nominal","target","heldout","retention"):
        rs = [r for r in rows if r["group"]==group]
        totals[group] = {"n":len(rs),"before":sum(r["before"]["pass"] for r in rs),"after":sum(r["after"]["pass"] for r in rs),
                         "lost":sum(r["before"]["pass"] and not r["after"]["pass"] for r in rs),
                         "gained":sum(not r["before"]["pass"] and r["after"]["pass"] for r in rs)}
    base,patch=compile_weights(spec),compile_weights(patched)
    # Does the edit stay quiet even on states reached by the modified healthy rollout?
    follow=torque_response(base,patch,[traces["nominal"]["after"]])
    h = torque_response(base,patch,hidden_healthy)
    r = torque_response(base,patch,hidden_failing,risky)
    n = torque_response(base,patch,hidden_failing,lambda s: not risky(s) and abs(s["pitch"])<.75)
    previous.write(out/"final-results.json", {"selected":pick["id"],"qualified_in_development":pick["qualified"],
                  "totals":totals,"rows":rows,"nominal_after_on_policy_response":follow,
                  "heldout_responses":{"healthy":h,"risk":r,"nonrisk_failure_trajectory":n}})
    print(json.dumps(totals))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("op",choices=["dev","lock","final"])
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--proposal",type=Path)
    p.add_argument("--round",type=int)
    a=p.parse_args()
    if a.op=="lock":
        lock(a.out)
    elif a.op=="dev":
        dev(previous.read(a.checkpoint),previous.read(a.proposal),a.round,a.out)
    else:
        final(previous.read(a.checkpoint),a.out)


if __name__=="__main__":
    main()
