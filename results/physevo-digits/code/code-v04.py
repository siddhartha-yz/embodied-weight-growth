import math

def decide(o):
    """Feedback-gated transfers; never infer held-object identity from jaw width."""
    s = dict(o.get('state') or {})
    arms, side = o['arms'], o['arm']
    last = o.get('last') or {}
    table = float(o['table_z'])

    def distance(a,b):
        return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

    def emit(op, arm=None, **kw):
        r = dict(op=op, state=s, **kw)
        if arm is not None:
            r['arm'] = arm
        return r

    def stop(reason):
        s['halted'] = reason
        return emit('fail_reobserve', reason=reason)

    def issue(op, arm, **kw):
        if op in ('move_tcp','pose_preview'):
            x,y,z = kw['xyz']
            if not all(math.isfinite(v) for v in [x,y,z,kw['yaw']]) or not (abs(x)<=.62 and -.45<=y<=.52 and table-.005<=z<=1.17):
                return stop('Invalid target; retain current grip and revise calibrated geometry.')
        p = dict(op=op, arm=arm, **kw)
        s['pending'] = p
        return dict(p, state=s)

    identity = [o['kind'],side,o.get('digit')]
    # Old recovery states assigned the requested digit to an already occupied
    # gripper. Their labels are not acquisition evidence. Preserve them for
    # inspection, but never migrate them into a trusted held-object transfer.
    if s and s.get('version') != 4:
        return stop('Legacy state lacks trustworthy held-object provenance. Inspect wrist RGB and reconcile the ACTUAL held object with its own destination and orientation. Do not open, relabel it as the requested digit, or advance the transfer queue.')
    if s.get('halted'):
        return emit('fail_reobserve',reason=s['halted'])
    if s and s['identity'] != identity:
        return stop('Transfer identity changed while state was active. Resolve the previous held object before starting another transfer.')

    if not s:
        s.update(version=4, identity=identity, stage='open', held=False,
                 acquired=False, tries=0)
        if o['kind']=='transfer':
            jaw = arms[side].get('jaw')
            if jaw is None or jaw < .85:
                s['occupancy_unknown'] = True
                return stop('Active gripper is not verified open at transfer start. A held object may belong to a PREVIOUS transfer. Inspect wrist RGB, identify it, and resolve its placement first. No opening, pickup, yaw scan, or carry toward this new destination is authorized.')
            src,dst = list(o['source']),list(o['dest'])
            a = (float(o['source_yaw'])+math.pi)%math.pi-math.pi/2
            turn = (float(o['turn_rad'])+math.pi)%(2*math.pi)-math.pi
            s.update(src=src,dst=dst,yaw=a,goal=a+turn,
                     height=max(table+.14,src[2]+.10,dst[2]+.10))
        elif o['kind']=='home':
            if any(arms[k].get('jaw',0)<.85 for k in ('left','right')):
                return stop('Homing would risk an unknown held object. Resolve occupied grippers before homing.')
            s.update(stage='home_clear',index=0)
        else:
            return stop('Unsupported operation kind.')

    if o['kind']=='transfer':
        if distance(s['src'],o['source'])>.002 or distance(s['dst'],o['dest'])>.002:
            return stop('Targets changed during an active transfer. Reconcile object identity and pending action; do not reinterpret a held silhouette as a tabletop pickup.')
        if s['held'] and not s['acquired']:
            return stop('Held-object provenance missing; request visual identity reconciliation.')
    if int(o.get('remaining_steps',0))<25:
        return stop('Low step reserve; keep current grip and request recovery.')

    p = s.get('pending')
    if p:
        m = arms[p['arm']]
        status = str(last.get('status') or '').lower()
        reason = str(last.get('reason') or '').lower()
        complete = status in ('completed','success','succeeded','ok')
        if last.get('op') not in (None,p['op']):
            return stop('Receipt mismatch; reconcile executor state.')
        if p['op']=='pose_preview':
            q = last.get('q_rad')
            if status in ('planned','completed','success','ok') and q is not None and len(q)==6 and all(math.isfinite(v) for v in q):
                cost = max(abs(x-y) for x,y in zip(q,m['q']))
                s['candidates'].append([cost,p['yaw']])
            s['preview_i'] += 1
            s.pop('pending',None)
        else:
            if p['op']=='move_tcp':
                err = distance(m['tcp'],p['xyz'])
                s['last_tcp_error'] = err
                tol = .004 if s['stage'] in ('down','place') else .008
                # XYZ alone cannot confirm a successful yaw command.
                ok = complete and err<=tol
            elif p['op']=='grip':
                jaw = m.get('jaw')
                if jaw is None:
                    return stop('Missing measured jaw after grip; inspect contact before continuing.')
                if complete and p['jaw']==0 and jaw<=.018:
                    return stop('Likely empty close: reobserve source center and contact. Do not transit.')
                ok = complete and (jaw>=.85 if p['jaw']==1 else .018<jaw<.8)
            else:
                err = max(abs(x-y) for x,y in zip(m['q'],p['q']))
                ok = err<=.025 and (complete or reason in ('execution_timeout','tracking_error'))
            if not ok:
                s['tries'] += 1
                retryable = complete or reason in ('execution_timeout','tracking_error','tracking error')
                if retryable and s['tries']<=2:
                    return dict(p,state=s)
                return stop('Action failed: '+reason+'. Check measured TCP error and jaw. Preserve held-object identity and grip; revise path or obtain an observed handoff. Do not advance to another digit.')
            s.pop('pending',None)
            s['tries'] = 0
            stage = s['stage']
            if stage=='close':
                s.update(held=True,acquired=True,acquisition_identity=list(identity),
                         acquisition_source=list(s['src']))
            elif stage=='release':
                s['held'] = False
            if stage=='rotate':
                shift = p['yaw']-s['next_yaw']
                s['goal'] += shift
                s['yaw'] = p['yaw']
                s['stage'] = 'rotation_check'
            elif stage!='home_joint':
                transitions = {'open':'clear','clear':'above','above':'down',
                    'down':'close','close':'lift','lift':'carry','carry':'rotation_check',
                    'place':'release','release':'retreat','retreat':'park','park':'done',
                    'home_clear':'home_park','home_park':'home_joint','home_open':'home_next'}
                if stage not in transitions:
                    return stop('Unexpected stage after receipt.')
                s['stage'] = transitions[stage]

    if o['kind']=='home':
        if s['stage']=='home_next':
            s['index'] += 1
            s['stage'] = 'home_clear'
        if s['index']>=2:
            if all(max(abs(x-y) for x,y in zip(arms[k]['q'],arms[k]['home_q']))<=.025 and arms[k].get('jaw',0)>=.85 for k in ('left','right')):
                return emit('done')
            return stop('Both arms must match measured home_q and have open jaws.')
        k = ('left','right')[s['index']]
        m = arms[k]
        h = max(table+.15,m['home_tcp'][2])
        if s['stage']=='home_clear':
            return issue('move_tcp',k,xyz=[m['tcp'][0],m['tcp'][1],max(h,m['tcp'][2])],yaw=0.)
        if s['stage']=='home_park':
            return issue('move_tcp',k,xyz=[m['home_tcp'][0],m['home_tcp'][1],h],yaw=0.)
        if s['stage']=='home_joint':
            delta = max(abs(x-y) for x,y in zip(m['q'],m['home_q']))
            if delta>.025:
                other = arms['right' if k=='left' else 'left']
                if m['tcp'][2]<table+.09 or distance(m['tcp'],other['tcp'])<.18:
                    return stop('Insufficient clearance for direct homing. Request collision-aware separation, preserving open empty jaws.')
                f = min(1.,.10/delta)
                return issue('joint',k,q=[x+f*(y-x) for x,y in zip(m['q'],m['home_q'])],steps=10)
            s['stage']='home_open'
        return issue('grip',k,jaw=1)

    if s['held'] and not (.018<arms[side].get('jaw',0)<.8):
        return stop('Grip changed during transport; inspect retention and identity before further motion.')
    st,h,dst = s['stage'],s['height'],s['dst']
    if st=='done':
        return emit('done')
    if st=='rotation_check':
        delta = s['goal']-s['yaw']
        if abs(delta)<.001:
            s['stage']='place'
            st='place'
        else:
            s.update(stage='rotation_preview',next_yaw=s['yaw']+max(-.15,min(.15,delta)),preview_i=0,candidates=[])
            st='rotation_preview'
    if st=='rotation_preview':
        i=s['preview_i']
        if i<3:
            return issue('pose_preview',side,xyz=[dst[0],dst[1],h],yaw=s['next_yaw']+(0,2*math.pi,-2*math.pi)[i])
        if not s['candidates']:
            return stop('No usable rotation preview. Keep grip and request a revised high-clearance waypoint.')
        cost,y = min(s['candidates'])
        if cost>.75:
            return stop('Rotation preview requires a large absolute joint change. Revise posture at clearance; do not bypass the planner.')
        s['stage']='rotate'
        return issue('move_tcp',side,xyz=[dst[0],dst[1],h],yaw=y)
    if st in ('open','close','release'):
        return issue('grip',side,jaw=0 if st=='close' else 1)
    if st=='clear':
        t=arms[side]['tcp']; xyz=[t[0],t[1],max(h,t[2])]
    elif st in ('above','lift'):
        xyz=[s['src'][0],s['src'][1],h]
    elif st=='down':
        xyz=s['src']
    elif st in ('carry','retreat'):
        xyz=[dst[0],dst[1],h]
    elif st=='place':
        xyz=dst
    elif st=='park':
        t=arms[side]['home_tcp']; xyz=[t[0],t[1],max(h,t[2])]
    else:
        return stop('Unexpected controller stage.')
    return issue('move_tcp',side,xyz=xyz,yaw=s['yaw'])
