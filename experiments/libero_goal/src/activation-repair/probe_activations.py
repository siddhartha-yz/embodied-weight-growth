"""Diagnose SmolVLA action-readout feature contributions, without changing deployed weights."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bp-control'))
from bp_fit import SmolActor, BowlOffsetEnv, cat_tree, denoise, gpu_tree, head, native_affine, replay, src_chunk

AXES = (1, 2)
VALIDATION = {
    'historical-bottle': (20, 40),
    'bowl-rim-dev': (50, 60),
    'clean-bottle-2': (50, 60),
    'clean-bowl-2': (50, 60),
}
DEFAULT_TRAINING = ROOT / '../runs/bp-control/training.pt'
STEP = .2


def compact_dump(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, allow_nan=False, separators=(',', ':'))+'\n')
    tmp.replace(path)


def split(row):
    if row['group'] != 'preserve':
        return 'fault-'+row['goal']
    return 'normal-carry' if row['goal'] in row['observation']['grasped'] else 'pregrasp'


def validation_contexts(actor, affine, gpu):
    rows = []
    with BowlOffsetEnv(pair='objects', camera_size=256, render_gpu_device_id=gpu) as env:
        for case, ticks in VALIDATION.items():
            d = ROOT / '../results/repair-series/replay-stock' / case
            result = json.loads((d/'result.json').read_text())
            trace = json.loads((d/'trace.json').read_text())
            requests = {t: {'id': f'{case}/activation-validation/{t}', 'group': 'preserve',
                'target': src_chunk(trace,t), 'axes': tuple(range(7))} for t in ticks}
            rows += replay(actor,env,result,trace,d/'initial-public.npz',requests,affine)
    return rows


def pred(actor, inp, affine):
    return denoise(actor, inp['prefix'], inp['noise'], affine)


@torch.no_grad()
def get_hidden_and_controls(actor, rows, affine):
    layer = head(actor)
    controls, hidden, inputs = {}, {}, {}
    for row in rows:
        name=row['id']
        inp=gpu_tree(row['input'], actor.device)
        captured=[]
        hook=layer.register_forward_pre_hook(lambda module,args: captured.append(args[0][:,:10].detach()))
        out=pred(actor,inp,affine)[0]
        hook.remove()
        if (out.cpu()-torch.tensor(row['stock'])).abs().max().item()>1e-6:
            raise ValueError(f'Fixed-input stock mismatch at {name}')
        # Euler sign; every denoise input is captured, not only the last hidden tensor.
        hidden[name] = -torch.cat(captured).mean(0).cpu().numpy()
        controls[name] = out.cpu().numpy()
        inputs[name] = inp
    return hidden,controls,inputs


def feature_shortlist(rows, hidden, affine, old):
    scales=affine[1].detach().cpu().numpy()[list(AXES)]
    # Frozen activation readout is only a cheap prefilter: the real pulse experiments follow.
    r={}
    den={}
    for group in ('fault-bottle','fault-bowl','normal-carry','pregrasp'):
        xs=[]; ys=[]
        for q in rows:
            if split(q)!=group: continue
            h=hidden[q['id']]
            xs.append(h[:,None,:]*scales[None,:,None])
            ys.append((np.array(q['target'])-q['stock'])[:,list(AXES)])
        x=np.concatenate(xs); y=np.concatenate(ys)
        den[group]=np.mean(x*x,axis=0)
        r[group]=np.mean(x*y[:,:,None],axis=0)
    guard=den['pregrasp']+.5*den['normal-carry']+1e-3
    score=lambda a, b: np.sum(a*a/(b+guard),axis=0)
    measures=[score(r['fault-bottle'],den['fault-bottle']),
              score(r['fault-bowl'],den['fault-bowl']),
              score(.5*(r['fault-bottle']+r['fault-bowl']),.5*(den['fault-bottle']+den['fault-bowl']))]
    found={}
    # Exclude the 8 previously selected columns; all selectable columns are new.
    for rank in range(20):
        for stat, m in zip(('bottle','bowl','joint'),measures):
            j=int(np.argsort(-m,kind='stable')[rank])
            if j not in old:
                found.setdefault(j,stat)
            if len(found)>=24: break
        if len(found)>=24: break
    return found


@torch.no_grad()
def pulse(actor, inp, affine, specs):
    """Intervene on the selected feature-to-output contribution on every denoise call."""
    B=len(specs)
    xcat=cat_tree([inp]*B)
    c=torch.tensor([v[1] for v in specs],device=actor.device)
    w=torch.zeros((B,7),device=actor.device)
    for i,(r,_,v) in enumerate(specs): w[i,r]=v
    def edit(module, args, output):
        # Equivalent to adding delta to one outgoing connection, but executed as an activation
        # intervention; no parameter edits or phase routing in this diagnostic.
        h=args[0].gather(2,c[:,None,None].expand(-1,args[0].shape[1],1))
        return torch.cat((output[:,:,:7]+h*w[:,None,:], output[:,:,7:]),dim=-1)
    h=head(actor).register_forward_hook(edit)
    try:
        return pred(actor,xcat,affine).cpu().numpy()
    finally:
        h.remove()


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def metrics(rows,effects,coord,sgn):
    p={k:[] for k in ('pregrasp','normal-carry','fault-bottle','fault-bowl')}
    signal={k:[] for k in ('fault-bottle','fault-bowl')}
    split_axes=list(AXES)
    for row in rows:
        k=split(row)
        d=effects[coord,sgn][row['id']]
        p[k].append(d)
        if k.startswith('fault'):
            residual=(np.array(row['target'])-row['stock'])[:,split_axes]
            dx=d[:,split_axes]
            signal[k].append((float(np.mean(residual*dx)),float(np.mean(dx*dx)),float(np.mean(residual*residual))))
    out={}
    for k, values in p.items():
        if not values: continue
        delta=np.stack(values)
        out[k]={'rms_7d':rms(delta), 'max_abs_7d':float(np.max(abs(delta))),
                'mean_yz':delta[:,:,[1,2]].mean(axis=(0,1)).tolist(),
                'per_context_rms':[rms(a) for a in delta]}
        if k in signal:
            align,norm,base=np.mean(signal[k],axis=0)
            out[k].update({'teacher_alignment':float(align),'teacher_mse_drop':float(2*align-norm),
                            'teacher_relative_mse_drop':float((2*align-norm)/base)})
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,default=DEFAULT_TRAINING)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--render-gpu',type=int,default=0)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    actor=SmolActor(n_action_steps=10)
    affine=native_affine(actor)
    dataset=torch.load(args.dataset,weights_only=True)
    discovery=dataset['contexts']
    validation=validation_contexts(actor,affine,args.render_gpu)
    hidden, controls, inputs=get_hidden_and_controls(actor, discovery+validation,affine)
    assert not set(q['id'] for q in discovery)&set(q['id'] for q in validation)
    old=set(x['column'] for x in dataset['coordinates'])
    found=feature_shortlist(discovery,hidden,affine,old)
    coords=[(r,c) for c in sorted(found) for r in AXES]
    # Same fixed prefix/noise is cloned within each batch. Every variant is subtracted from a
    # zero-pulse sibling in the same batch to cancel BF16 batch-shape drift.
    specs=[(r,c,s) for r,c in coords for s in (1,-1)]
    all_effects={((r,c),s):{} for r,c,s in specs}
    batch_error=0.
    for n,row in enumerate(discovery+validation):
        key=row['id']; inp=inputs[key]
        for start in range(0,len(specs),args.batch-1):
            chunk=specs[start:start+args.batch-1]
            variants=pulse(actor,inp,affine,[(1,0,0)]+[(r,c,STEP*s) for r,c,s in chunk])
            batch_error=max(batch_error,float(np.max(abs(variants[0]-controls[key]))))
            for i,(r,c,s) in enumerate(chunk):
                all_effects[(r,c),s][key]=variants[i+1]-variants[0]
        print(f'Context {n+1:2}/{len(discovery)+len(validation)} {key} done',flush=True)
    summaries={}
    for r,c in coords:
        summaries[f'{r}/{c}']={'row':r,'column':c,'stock':float(head(actor).weight[r,c]),
            'screen':found[c],
            'signed':{str(s):{'train':metrics(discovery,all_effects,(r,c),s),
                              'development_check':metrics(validation,all_effects,(r,c),s)} for s in (1,-1)}}
    # Exact single-context confirmation of the strongest positive measured repair-vs-leakage pulses
    # across all phases and independent development times; report rather than selecting on those checks.
    def select_key(pair):
        coord,s=pair
        train=summaries[f'{coord[0]}/{coord[1]}']['signed'][str(s)]['train']
        gain=sum(train[k]['teacher_relative_mse_drop'] for k in ('fault-bottle','fault-bowl'))/2
        return gain/max(train['pregrasp']['rms_7d'],.005)
    top=[]
    seen=set()
    for coord,s in sorted(all_effects,key=select_key,reverse=True):
        if coord in seen: continue
        top.append((coord,s)); seen.add(coord)
        if len(top)==8: break
    confirmations=[]
    for coord,s in top:
        effects={(coord,s):{}}
        for q in discovery+validation:
            key=q['id']; inp=inputs[key]
            got=pulse(actor,inp,affine,[(coord[0],coord[1],STEP*s)])[0]
            effects[coord,s][key]=got-controls[key]
        confirmations.append({'row':coord[0],'column':coord[1],'offset':STEP*s,
            'train':metrics(discovery,effects,coord,s),'development_check':metrics(validation,effects,coord,s)})
        print(f'Single-context confirmation {coord} ({s:+d})',flush=True)
    ctx=lambda rows:[{k:v for k,v in q.items() if k!='input'}|{'phase':split(q)} for q in rows]
    compact={
        'experiment':'Readout activation contribution pulse, every denoise call. Same prefix, noise, and stock model.',
        'schema':'position [row,column], Y/Z are rows 1/2. Positive/negative readout pulse scale = ±0.2. '
                 'Full actions are 10 native commands with XYZ, RPY, gripper. RMS is native all-seven-axis RMS.',
        'method':'Only selectable columns are beyond prior 16 values. Frozen hidden readout prefilter on fit data, '
                 'then full sampler signed intervention. Single-context verification of strongest nonduplicate pulses.',
        'batch_control_max_native_difference_from_single':batch_error,
        'coordinates':[{k:v for k,v in q.items() if k in ('row','column','stock')} for q in summaries.values()],
        'discovery_contexts': [{'id':q['id'],'phase':split(q)} for q in discovery],
        'validation_contexts': [{'id':q['id'],'phase':split(q)} for q in validation],
        'signed_interventions':list(summaries.values()),
        'single_context_confirmation':confirmations,
    }
    full={**compact,'contexts':ctx(discovery),
          'objective':{'group_weights':{'repair-bottle':.5,'repair-bowl':.5,'preserve':1.},
                       'bound_from_stock':.65},
          'probes':[{'row':r,'column':c,'stock':float(head(actor).weight[r,c]),'step':STEP,
                     'effects':{q['id']:np.round(all_effects[(r,c),1][q['id']].astype(np.float64),6).tolist()
                                for q in discovery}}
                    for r,c in coords]}
    compact_dump(args.out/'localization-compact.json',compact)
    compact_dump(args.out/'localization.json',full)
    print(f'Locked {len(found)} selectable columns / {len(coords)} coordinates',flush=True)

if __name__=='__main__': main()
