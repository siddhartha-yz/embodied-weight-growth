"""Ask the established Codex SDK transport for a bowl-only numerical SmolVLA patch."""
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

ASK=Path(__file__).resolve().parents[1]/"codex_transport.py"
spec=importlib.util.spec_from_file_location("established_astra_sdk",ASK)
transport=importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
obj=transport.object_schema
transport.EFFORT="medium"
transport.BASE=("Diagnose a pretrained SmolVLA based on supplied observations and actual forward probes. "
                "Use only the evidence. Write direct scalar deltas, not a controller, code or training.")
transport.SCHEMA=obj({"edits":{"type":"array","items":obj({
 "row":{"type":"integer"},"column":{"type":"integer"},"delta":{"type":"number"}})},
 "reason":{"type":"string"}})
transport.parse_answer=json.loads

def main():
    probes, out = map(Path,sys.argv[1:3])
    data=json.loads(probes.read_text())
    for k in ("fault_trace","clean_trace"):
        rows=data[k]; end=data["fault"]["last_contact"]+20 if k=="fault_trace" else data["clean"]["first_native_goal"]+5
        data[k]=[r for r in rows if r["t"]<=end]
    prompt="""Propose one direct numerical action-head patch for BOWL ONLY from the original SmolVLA checkpoint. No bottle patch or held-out results are supplied. Use the confirmed bowl failure and matched successful original-layout bowl. Each candidate runs fixed from step 1 with the same native text and RGB/proprioception, so preserving grasp acquisition is important. Do not add a runtime gate, task ID, trigger, rescue script, training, or BP. Read the trace and diagnose which closed-loop command differences would actually help before choosing edits. Concentrate on changes active around the problem and small on both early grasp and successful transport.

On the actual stock trace the bowl was held at step 110, still 38 mm outside the top footprint, while Z was already being commanded down. At step 118 the bowl hit the cabinet top edge, was about 10 mm short in XY, and lost two-finger contact while the gripper was still commanded closed. It did not command open until step 121. Thus do not diagnose the known failure as early opening; consider delayed descent or enough inward travel.

Edit existing FP32 parameters of model.action_out_proj (shape [32,720]), up to 8 different measured coefficients, finite |delta|<=.65. Output rows 0,1,2,6 are native X/Y/Z and gripper (positive = close). In this flow-matching model a positive parameter change does not generally imply a positive native movement. Use measured responses and preserve the correct early grasp. Each probe changes one scalar by +.30 under the exact same observation/prompt/noise. Effects are [mean native delta in edited action axis for first 5 and last 5 controls] in its replanned 10-control chunk. Mixed edits and closed-loop trajectories need not be linear. Context means are [x,y,z,gripper]. The rim metric is native physical bowl contact with a cabinet geom, computed host-side; it is not exposed as an actor input. Only original and bowl-failure development evidence, not the final blind evaluation, is below.
"""
    prompt+="\nDevelopment evidence: "+json.dumps(data,separators=(",",":"))
    if len(sys.argv)>3:
        feedback=Path(sys.argv[3]).read_text()
        prompt+="\nOne tested development-only repair and its outcomes, not held-out: "+feedback
        prompt+="\nReturn a full replacement for that patch (relative to original model, not an increment to it)."
    out.with_suffix(".prompt.txt").write_text(prompt)
    sdk=transport.import_sdk()
    audit={"model_requested":transport.MODEL,"reasoning_effort":transport.EFFORT,"answer":None}
    start=perf_counter()
    try:
        audit["answer"]=transport.ask(sdk,prompt,[],audit)
        out.write_text(json.dumps({"id":"bowl",**audit["answer"]},indent=2)+"\n")
    except Exception as exc:
        audit["error"]=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        audit["wall_s"]=perf_counter()-start
        out.with_suffix(".audit.json").write_text(json.dumps(audit,indent=2)+"\n")
if __name__=="__main__": main()
