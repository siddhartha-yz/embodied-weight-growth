"""One text-only SDK turn; install only the numeric answers in the simulator."""
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

ASK=Path(__file__).resolve().parents[1]/"codex_transport.py"
spec = importlib.util.spec_from_file_location("established_astra_sdk", ASK)
transport=importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
obj=transport.object_schema
transport.BASE = ("You are diagnosing an existing SmolVLA action expert. Use only supplied data. "
                  "Return direct finite scalar edits, not runtime logic, code, training, or tests. "
                  "The environment will execute your proposed numeric coefficients.")
transport.EFFORT = "medium"
transport.SCHEMA = obj({"candidates":{"type":"array","items":obj({
    "id":{"type":"string"},
    "edits":{"type":"array","items":obj({"row":{"type":"integer"},"column":{"type":"integer"},"delta":{"type":"number"}})},
    "reason":{"type":"string"},
})}})
transport.parse_answer=json.loads

def context(data, episodes):
    base={}
    for k,v in data["stock_chunks"].items():
        base[k]={"xyz_first5":v["xyz_first5_mean"],"xyz_last5":v["xyz_last5_mean"],"g10":v["gripper"]}
    measured=[]
    for p in data["probes"]:
        e=p["edit"]; r=e["row"]
        by={}
        for name,val in p["chunks"].items():
            by[name]=[val["first5_mean_delta_in_edited_axis"],val["last5_mean_delta_in_edited_axis"],val["unedited_rms"]]
            if r==6 and name.startswith("fault"): by[name].append(val["gripper_chunk"])
        measured.append([e["row"],e["column"],e["delta"],round(p["old"],5),by])
    result={}
    for name,x in episodes.items():
        result[name]={k:x[k] for k in ("success","first_lift","first_lift_step","first_open_after_lift","last_two_pad_contact","first_intended_goal_step")}
        result[name]["obj_path"]={str(t):[round(a,3) for a in d["target_xyz"]] for t,d in x["milestones"].items()}
    return {"shape":data["shape"],"episodes":result,"stock":base,"one-edit-probes":measured}

def main():
    evidence, episodes, out = map(Path,sys.argv[1:])
    data=json.loads(evidence.read_text())
    prompt="""Choose two distinct candidate numeric patches (one prioritizes preservation and one more assertive, at most 8 scalars each) for the pretrained SmolVLA action output linear projection. Do not write a controller. IDs should be conservative and assertive.

Robot already acquired the correct wine bottle by control 70 after a fixed prompt-only acquisition. At control 81 onward it sees the exact native instruction, identical in all evaluation arms. In the crossed layout the model moves toward the cabinet, stalls around bottle (x=+.021,y=-.155,z=1.132)m, opens at control 106 and loses grasp at 107. The correct canonical model placed by step90 while still commanding close; native score is support on cabinet, not a release command. Canonical successes are within x[-.015,+.030], y[-.229,-.199], z[1.129,1.134] m. There is no need to force a specific x or path. It may need lift clearance or longer negative-y motion while retaining the bottle; do not disrupt acquisition or canonical bowl/bottle. The hold and carry rows in evidence are diagnostic downstream action clamps, not allowed in your patches or the deployment.

The same set of direct scalars will be active from control 81 on fault (the previous unchanged policy gets it into the exact same held state). We also deploy each patch from control 1 in canonical controls to detect side effects. No task-stage gate, timing code, extra inference hooks, training, or BP is allowed. You may use any of the measured columns (or bias column -1). Rows 0,1,2,6 correspond to native x,y,z,gripper. Positive native gripper = close, negative = open. Positive y = away from destination. Use finite |delta|<=0.65 for weights, <=1.25 for biases. Forward Euler integration of predicted velocity runs backwards; use *measured* sensitivities below rather than assuming coefficient sign equals movement. Probe data is one edit at a time on identical live observation and noise; combining and rolling out can be nonlinear.

Each measured row is [projection row, column(-1=bias), tested delta, old weight, per-context [mean action delta over first five controls on affected axis, mean over last five, RMS delta in other axes, (only on gripper in fault: absolute gripper sequence)]]. A context suffix @100 means after 100 executed controls, before predicting controls 101..110. Stock summary xyz lists the means over each half and g10 is the 10-action gripper sequence. Clean control windows differ semantically; preserve good action signals and don't simply saturate gripper close everywhere.
"""
    prompt+="\nEvidence JSON: "+json.dumps(context(data,json.loads(episodes.read_text())),separators=(",",":"))
    (out.parent/"prompt.txt").write_text(prompt)
    sdk=transport.import_sdk()
    audit={"model_requested":transport.MODEL,"reasoning_effort":transport.EFFORT,"answer":None}
    start=perf_counter()
    try:
        audit["answer"]=transport.ask(sdk,prompt,[],audit)
        c=audit["answer"]["candidates"]
        assert [x["id"] for x in c]==["conservative","assertive"]
        for x in c:
            (out.parent/(x["id"]+".json")).write_text(json.dumps(x,indent=2)+"\n")
    except Exception as exc:
        audit["error"]=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        audit["wall_s"]=perf_counter()-start
        out.write_text(json.dumps(audit,indent=2)+"\n")
if __name__=="__main__": main()
