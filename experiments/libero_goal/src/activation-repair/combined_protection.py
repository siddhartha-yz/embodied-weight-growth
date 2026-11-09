"""Measure locked full edits' same-input action leakage on eight extra development contexts."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from probe_activations import (SmolActor, compact_dump, gpu_tree, head, native_affine,
                               pred, split, validation_contexts)

ARMS = ('auto-bp', 'astra-bp', 'astra-wsp')


def read_manifest(path, deadline):
    while True:
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                # A publisher may still be writing the locked file.
                pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Waiting for locked edit manifest: {path}')
        time.sleep(2)


@torch.no_grad()
def measure(actor, rows, inputs, stock, affine, manifest, allowed):
    layer = head(actor)
    edits = manifest['edits']
    coordinates = [(e['row'], e['column']) for e in edits]
    if len(coordinates) > 16 or len(set(coordinates)) != len(coordinates) or not set(coordinates) <= allowed:
        raise ValueError(f'Expected at most 16 unique locked readout coordinates; got {coordinates}')
    changes = []
    baseline = layer.weight.detach().clone()
    try:
        for e, (r, c) in zip(edits, coordinates):
            v = float(e['value'])
            if not math.isfinite(v) or abs(v-float(baseline[r,c])) > .650001:
                raise ValueError(f'Out-of-bound edit {(r,c,v)}')
            layer.weight[r,c] = v
        for row in rows:
            key=row['id']
            arr=(pred(actor, inputs[key], affine)[0].cpu().numpy()-stock[key]).astype(np.float64)
            changes.append({'id':key, 'phase':split(row), 'rms_7d':float(np.sqrt(np.mean(arr*arr))),
                            'max_abs_7d':float(np.max(abs(arr))), 'action_shift':arr.tolist()})
    finally:
        layer.weight.copy_(baseline)
    summaries = {}
    for group in ('pregrasp','normal-carry'):
        use = [x for x in changes if x['phase']==group]
        delta = np.array([x['action_shift'] for x in use])
        summaries[group] = {'contexts':len(use), 'rms_7d':float(np.sqrt(np.mean(delta*delta))),
            'max_abs_7d':max(x['max_abs_7d'] for x in use),
            'max_window_rms_7d':max(x['rms_7d'] for x in use),
            'axis_rms':np.sqrt(np.mean(delta*delta,axis=(0,1))).tolist()}
    return {'coordinates':coordinates, 'phases':summaries, 'per_context':changes}


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--render-gpu',type=int,default=0)
    p.add_argument('--wait-seconds',type=int,default=600)
    args=p.parse_args()
    loc=json.loads((args.run/'localization.json').read_text())
    allowed={(q['row'],q['column']) for q in loc['coordinates']}
    actor=SmolActor(n_action_steps=10)
    affine=native_affine(actor)
    rows=validation_contexts(actor,affine,args.render_gpu)
    expected={q['id']:q['phase'] for q in loc['validation_contexts']}
    assert {q['id']:split(q) for q in rows}==expected, 'Must use identical disjoint development times'
    inputs={q['id']:gpu_tree(q['input'],actor.device) for q in rows}
    stock={key:pred(actor,inp,affine)[0].cpu().numpy() for key,inp in inputs.items()}
    err=max(float(np.max(abs(stock[q['id']]-np.asarray(q['stock'])))) for q in rows)
    assert err<=1e-6, f'Stock fixed-input max error {err}'
    print(f'Verified exact baseline on {len(rows)} contexts; waiting for final edit manifests',flush=True)
    deadline=time.monotonic()+args.wait_seconds
    result={'stock_max_replay_error':err,
            'condition':'same eight extra development contexts and single-context action sampler/noise as localization',
            'axes':['x','y','z','rx','ry','rz','gripper'],'arms':{}}
    for arm in ARMS:
        m=read_manifest(args.run/f'{arm}.json',deadline)
        result['arms'][arm]=measure(actor,rows,inputs,stock,affine,m,allowed)
        print(arm,json.dumps(result['arms'][arm]['phases'],separators=(',',':')),flush=True)
    compact_dump(args.run/'combined-protection.json',result)


if __name__=='__main__':
    main()
