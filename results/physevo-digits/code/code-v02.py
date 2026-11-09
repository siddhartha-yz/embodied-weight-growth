import math

def decide(o):
    s = dict(o.get('state') or {})
    arms, arm = o['arms'], o['arm']
    last = o.get('last') or {}
    table = float(o['table_z'])

    def distance(a, b):
        return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

    def out(op, side=None, **kw):
        r = dict(op=op, state=s, **kw)
        if side is not None:
            r['arm'] = side
        return r

    def stop(reason):
        s['halted'] = reason
        return out('fail_reobserve', reason=reason)

    def issue(op, side, **kw):
        if op in ('move_tcp', 'pose_preview'):
            x,y,z = kw['xyz']
            if not all(math.isfinite(v) for v in [x,y,z,kw['yaw']]) or not (abs(x)<=.62 and -.45<=y<=.52 and table-.005<=z<=1.17):
                return stop('Invalid target bounds; retain grip and revise calibrated geometry.')
        p = dict(op=op, arm=side, **kw)
        s['pending'] = p
        return dict(p, state=s)

    # Migrate the previous controller only at its documented held-object
    # rotation failure. Never reinterpret the projected held silhouette as
    # a new tabletop pickup or as a new wrist-yaw measurement.
    if s and s.get('version') != 2:
        if o['kind']=='transfer' and s.get('held') and s.get('phase')==6 and s.get('signature', [None,None])[1]==arm:
            old = s
            s = dict(version=2, kind='transfer', arm=arm, digit=o.get('digit'),
                     stage='carry', held=True, yaw=old['yaw'],
                     goal_yaw=old['final_yaw'], dst=list(o['dest']),
                     height=max(old['height'],table+.14), tries=0,
                     recovery_note='Retained last confirmed yaw and original signed turn; new zero turn was visually uncertain.')
        else:
            return stop('Cannot safely migrate this state; reconcile pending motion and held object first.')

    if not s:
        s.update(version=2, kind=o['kind'], arm=arm, digit=o.get('digit'), tries=0, held=False)
        if o['kind']=='home':
            if any(arms[k].get('jaw',0)<.85 for k in ('left','right')):
                return stop('Homing requires known-empty open jaws; inspect before opening an unknown grip.')
            s.update(stage='home_clear', home_index=0)
        elif o['kind']=='transfer':
            if arms[arm].get('jaw',0)<.8:
                return stop('Possible held object with no history. Restore prior state containing last confirmed yaw; do not open or use projected source as tabletop pickup.')
            a = (float(o['source_yaw'])+math.pi) % math.pi-math.pi/2
            turn = (float(o['turn_rad'])+math.pi)%(2*math.pi)-math.pi
            src,dst = list(o['source']),list(o['dest'])
            s.update(stage='open', src=src, dst=dst, yaw=a, goal_yaw=a+turn,
                     height=max(table+.14,src[2]+.10,dst[2]+.10))
        else:
            return stop('Unknown operation kind.')

    if s.get('halted'):
        return out('fail_reobserve', reason=s['halted'])
    if s['kind']!=o['kind'] or s['arm']!=arm or s.get('digit')!=o.get('digit'):
        return stop('State identity mismatch; reconcile current object before starting another operation.')
    if o['kind']=='transfer' and distance(s['dst'],o['dest'])>.002:
        return stop('Destination changed during execution; reconcile held-object route explicitly.')
    if int(o.get('remaining_steps',0))<25:
        return stop('Low step reserve; retain present grip and request recovery.')

    p = s.get('pending')
    if p:
        m = arms[p['arm']]
        status = str(last.get('status') or '').lower()
        reason = str(last.get('reason') or '').lower()
        completed = status in ('completed','success','succeeded','ok')
        if last.get('op') not in (None,p['op']):
            return stop('Mismatched receipt; reconcile last executed operation.')
        if p['op']=='pose_preview':
            q = last.get('q_rad')
            if q is not None and len(q)==6 and all(math.isfinite(v) for v in q) and not any(w in status for w in ('fail','error','reject')):
                # Compare absolute joints, not wrapped joints: a 2*pi jump
                # can be a genuine limit-crossing posture change.
                cost = max(abs(x-y) for x,y in zip(q,m['q']))
                s['candidates'].append([cost,p['yaw']])
            s.pop('pending',None)
            s['preview_index'] += 1
        else:
            if p['op']=='move_tcp':
                tol = .004 if s['stage'] in ('down','place') else .008
                err = distance(m['tcp'],p['xyz'])
                s['last_tcp_error'] = err
                ok = completed and err<=tol
            elif p['op']=='grip':
                jaw = m.get('jaw')
                if jaw is None:
                    return stop('Measured jaw missing after grip; inspect contact.')
                ok = completed and (jaw>=.85 if p['jaw']==1 else .018<jaw<.8)
                if p['jaw']==0 and completed and jaw<=.018:
                    return stop('Likely empty close; reobserve grasp center before lifting.')
            else:
                ok = max(abs(x-y) for x,y in zip(m['q'],p['q']))<=.025
                ok = ok and (completed or reason in ('execution_timeout','tracking_error'))
            if not ok:
                s['tries'] += 1
                # A timeout at the requested XYZ does not establish yaw.
                # One repeat obtains a completion receipt, as in the supplied run.
                retryable = reason in ('execution_timeout','tracking_error','tracking error') or completed
                if retryable and s['tries']<=2:
                    return dict(p,state=s)
                return stop('Motion/grip did not converge: '+reason+'. Inspect measured TCP error and jaw. Keep held object closed; revise waypoint or observed handoff rather than repeat an infeasible path.')
            s.pop('pending',None)
            s['tries']=0
            st = s['stage']
            if p['op']=='grip':
                s['held'] = p['jaw']==0
            if st=='rotate':
                s['yaw']=s.pop('next_yaw')
                s['stage']='rotation_check'
            elif st=='home_joint':
                pass
            else:
                nxt = {'open':'clear','clear':'above','above':'down','down':'close',
                       'close':'lift','lift':'carry','carry':'rotation_check',
                       'place':'release','release':'retreat','retreat':'park','park':'done',
                       'home_clear':'home_park','home_park':'home_joint','home_open':'home_next'}
                s['stage']=nxt[st]

    if o['kind']=='home':
        idx=s['home_index']
        if s['stage']=='home_next':
            idx+=1
            s.update(home_index=idx,stage='home_clear')
        if idx>=2:
            for k in ('left','right'):
                m=arms[k]
                if max(abs(x-y) for x,y in zip(m['q'],m['home_q']))>.025 or m.get('jaw',0)<.85:
                    return stop('Both measured home joint vectors and open jaws are required.')
            return out('done')
        side=('left','right')[idx]
        m=arms[side]
        h=max(table+.15,m['home_tcp'][2])
        if s['stage']=='home_clear':
            return issue('move_tcp',side,xyz=[m['tcp'][0],m['tcp'][1],max(h,m['tcp'][2])],yaw=0.)
        if s['stage']=='home_park':
            return issue('move_tcp',side,xyz=[m['home_tcp'][0],m['home_tcp'][1],h],yaw=0.)
        if s['stage']=='home_joint':
            delta=max(abs(x-y) for x,y in zip(m['q'],m['home_q']))
            if delta<=.025:
                s['stage']='home_open'
            else:
                other=arms['right' if side=='left' else 'left']
                if m['tcp'][2]<table+.09 or distance(m['tcp'],other['tcp'])<.18:
                    return stop('Unsafe proximity/height for direct joint homing; request collision-aware recovery.')
                f=min(1.,.10/delta)
                return issue('joint',side,q=[x+f*(y-x) for x,y in zip(m['q'],m['home_q'])],steps=10)
        return issue('grip',side,jaw=1)

    if s['held'] and not (.018<arms[arm].get('jaw',0)<.8):
        return stop('Grip changed during transport; inspect object before further motion. Do not release.')
    st=s['stage']; h=s['height']; a=s['yaw']; dst=s['dst']
    if st=='done':
        return out('done')
    if st=='rotation_check':
        delta=s['goal_yaw']-a
        if abs(delta)<.001:
            s['stage']='place'; st='place'
        else:
            s.update(stage='preview', next_yaw=a+max(-.15,min(.15,delta)),preview_index=0,candidates=[])
            st='preview'
    if st=='preview':
        i=s['preview_index']
        if i<3:
            y=s['next_yaw']+(0,2*math.pi,-2*math.pi)[i]
            return issue('pose_preview',arm,xyz=[dst[0],dst[1],h],yaw=y)
        if not s['candidates']:
            return stop('No preview candidate for incremental held rotation; retain grip and request a revised high waypoint.')
        cost,y=min(s['candidates'])
        if cost>.75:
            return stop('Rotation preview requires a large absolute joint jump; retain grip and revise posture at clearance.')
        s['stage']='rotate'
        return issue('move_tcp',arm,xyz=[dst[0],dst[1],h],yaw=y)
    if st in ('open','close','release'):
        return issue('grip',arm,jaw=0 if st=='close' else 1)
    if st=='clear':
        t=arms[arm]['tcp']; xyz=[t[0],t[1],max(h,t[2])]
    elif st in ('above','lift'):
        xyz=[s['src'][0],s['src'][1],h]
    elif st=='down':
        xyz=s['src']
    elif st in ('carry','retreat'):
        xyz=[dst[0],dst[1],h]
    elif st=='place':
        xyz=dst
    elif st=='park':
        t=arms[arm]['home_tcp']; xyz=[t[0],t[1],max(h,t[2])]
    else:
        return stop('Unexpected controller stage; inspect persistent state.')
    return issue('move_tcp',arm,xyz=xyz,yaw=a)
