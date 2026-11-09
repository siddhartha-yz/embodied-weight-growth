"""One Astra editing opportunity; issue absolute values on the immediately preceding model."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
from time import perf_counter

SDK_HELPER=Path(__file__).resolve().parents[1]/"codex_transport.py"
spec=importlib.util.spec_from_file_location("established_astra_sdk",SDK_HELPER)
transport=importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
obj=transport.object_schema
transport.EFFORT="high"
transport.BASE=("You are the numerical weight editor in a closed-loop SmolVLA experiment. "
                "Use only the supplied probe evidence. Return direct absolute scalar values; no code, tools or training.")
transport.SCHEMA=obj({"edits":{"type":"array","items":obj({"row":{"type":"integer"},
       "column":{"type":"integer"},"value":{"type":"number"}})},"diagnosis":{"type":"string"}})
transport.parse_answer=json.loads

COMMON="""You have exactly one edit opportunity in this stage. Start from the currently evaluated model,
not a reset and not a weighted average of different models. Use at most 8 distinct tested FP32
coefficients in model.action_out_proj.weight, shape [32,720]. Set absolute values (not deltas);
each may differ from its measured current value by at most 0.65. Unlisted weights stay exactly as
in the current model. No new weights, controllers, instructions, routing, triggers, per-task
switches, gradients or tools. The edited model runs with fixed weights from control 1.

Rollouts are 20 Hz and replan in chunks of 10 controls. Positive gripper action closes.
The native goal predicate can briefly fire while an object is falling. The main target is a
true release onto the cabinet with the native goal holding for 40 consecutive ungrasped controls.
A native hit without stable release is only diagnostic. A useful fix must keep acquisition
of the right object and preserve already-working ordinary layouts.

Probes used actual observations and identical flow-matching noise from the current model's rollouts.
They change one scalar by +0.30 relative to its current absolute value. Effects and current actions
are [mean of first five controls, mean of last five controls], each in order X/Y/Z/gripper,
in the original native action coordinates. Row 0/1/2/6 correspond to those channels, but
the final native action can change on multiple axes and in a different direction than a weight
perturbation. Use measured effects (not the sign of a coefficient alone), prioritize the most
selective coefficients, and avoid large side effects in actual acquisition and clean-placement
contexts. Local responses are not guarantees for combined weights or closed-loop trajectories.
In traces xyz/eef are world meters, t is the control index, held is two-finger object contact.
"""
STAGES={
    1:"""Round 1/3. Repair WINE BOTTLE ONLY. You have one difficult bottle layout and one successful
ordinary bottle layout from the original model. In the difficult one, the instruction schedule is
a fixed early pick-then-place phrasing through control 80, then native phrasing for the rest.
The weight edit, unlike the instruction schedule, is in force from the start. A past intervention
that helped only when installed after grasp is insufficient; protect acquisition from step one.""",
    2:"""Round 2/3. Repair BOWL using the model after round 1. Round 1's assignments are already
in these weights and in every intervention below. Keep changes targeted, avoid disrupting ordinary
movement. No bottle rollout evidence is supplied in this round. You may change already edited
coefficients if the current bowl evidence warrants it; there is no requirement to add new ones.""",
    3:"""Round 3/3. Jointly repair BOWL and WINE BOTTLE. The latest detailed evidence and probes
below are from the single model after two edits. Earlier compact development outcomes show how
the same cases evolved; they are not alternative networks to add. You can revise or undo earlier
values where their measured effect conflicts with either task. Choose one shared set of values.
Aim for both difficult layouts with stable release, then protect ordinary bottle/bowl behavior.
Do not prioritize one target at the expense of all ability on the other. The same-input
stock-minus-current measurements compare actual stock weights with the current model at the
exact same observations and random noise. Earlier acquisition loss can be more decisive than
a hypothetical improvement to later placement. No more revisions.""",
}

def load(p):
    return json.loads(p.read_text())

def dump(p,obj):
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")

def fp32(x):
    return struct.unpack("f",struct.pack("f",x))[0]

def key(e):
    return e["row"],e["column"]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",type=int,choices=(1,2,3))
    p.add_argument("probes",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--previous",type=Path)
    p.add_argument("--history",type=Path,nargs="*")
    a=p.parse_args()
    if a.out.exists() or a.out.with_suffix(".audit.json").exists():
        p.error("An opportunity already has an answer/audit at this path")
    data=load(a.probes)
    prior=load(a.previous)["edits"] if a.previous else []
    assert (a.stage==1)==(a.previous is None)
    prompt=STAGES[a.stage]+"\n\n"+COMMON
    if prior:
        prompt+="\nCurrent accumulated absolute assignments (already installed): "+json.dumps(prior,separators=(",",":"))
    if a.history:
        fields=("id","first_lift","first_lift_step","first_cabinet_contact_after_lift","last_contact",
                "first_native_goal","stable_released_at","final_native")
        summaries={path.parent.name:[{k:row[k] for k in fields} for row in load(path)] for path in a.history}
        prompt+="\nEarlier development outcomes only: "+json.dumps(summaries,separators=(",",":"))
    prompt+="\nCurrent development evidence: "+json.dumps(data,separators=(",",":"))
    a.out.with_suffix(".prompt.txt").write_text(prompt)
    sdk=transport.import_sdk()
    audit={"stage":a.stage,"model_requested":transport.MODEL,"effort":transport.EFFORT,"answer":None}
    start=perf_counter()
    try:
        answer=transport.ask(sdk,prompt,[],audit)
        audit["answer"]=answer
        table={key(e):e["current"] for e in data["edits_tested"]}
        seen=set()
        current={key(e):e for e in prior}
        edits=answer["edits"]
        if len(edits)>8: raise ValueError("More than 8 weight assignments")
        for e in edits:
            k,v=key(e),e["value"]
            if k in seen or k not in table or type(v) not in (int,float) or not math.isfinite(v):
                raise ValueError(f"Invalid, duplicate or unprobed value: {e}")
            if abs(v-table[k])>.650001:
                raise ValueError(f"Value differs from current model by more than .65: {e}")
            seen.add(k)
            current[k]={"row":k[0],"column":k[1],"value":fp32(v)}
        dump(a.out,{"stage":a.stage,"edits":[current[k] for k in sorted(current)],
                    "round_assignments":edits,"diagnosis":answer["diagnosis"]})
        print(f"Round {a.stage}: {len(edits)} assignments; {len(current)} accumulated coordinates",flush=True)
    except Exception as exc:
        audit["error"]=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        audit["wall_s"]=perf_counter()-start
        dump(a.out.with_suffix(".audit.json"),audit)

if __name__=="__main__":
    main()
