import math

def decide(o):
    s = dict(o.get('state') or {})
    arms, side = o['arms'], o['arm']
    last = o.get('last') or {}
    tz = float(o['table_z'])

    def dist(a,b):
        return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

    def result(op, arm=None, **kw):
        r = dict(op=op, state=s, **kw)
        if arm is not None:
            r['arm']=arm
        return r

    def fail(message):
        s['halted']=message
        return result('fail_reobserve', reason=message)

    def issue(op, arm, **kw):
        if op in ('move_tcp','pose_preview'):
            x,y,z=kw['xyz']
            if not all(math.isfinite(v) for v in [x,y,z,kw['yaw']]) or not (abs(x)<=.62 and -.45<=y<=.52 and tz-.005<=z<=1.17):
                return fail('Invalid motion target; retain current grip and revise geometry.')
        p=dict(op=op,arm=arm,**kw)
        s['pending']=p
        return dict(p,state=s)

    def start_scan(center=None):
        # Inverse-query the public pose solver, not an invented FK model.
        s['scan_xyz']=list(arms[side]['tcp'])
        s['scan_q']=list(arms[side]['q'])
        s['scan_i']=0
        s['candidates']=[]
        if center is None:
            s['angles']=[-math.pi+i*math.pi/8 for i in range(16)]
            s['stage']='scan_coarse'
        else:
            s['angles']=[center+i*math.pi/32 for i in range(-2,3)]
            s['stage']='scan_fine'

    # The preceding attempt supplied no useful persistent grasp history.
    # Permit migration of that initialization-only refusal, not arbitrary failures.
    if s and s.get('version')!=3:
        if s.get('version')==2 and not s.get('pending') and 'no history' in s.get('halted','').lower():
            s={}
        else:
            return fail('Unsupported prior state: preserve grasp history and reconcile pending action before migration.')

    if not s:
        s.update(version=3,kind=o['kind'],arm=side,digit=o.get('digit'),tries=0,held=False)
        if o['kind']=='home':
            if any(arms[k].get('jaw',0)<.85 for k in ('left','right')):
                return fail('Unknown closed gripper during homing: verify empty before opening.')
            s.update(stage='home_clear',index=0)
        elif o['kind']=='transfer':
            dst=list(o['dest'])
            s.update(dst=dst,height=max(tz+.14,dst[2]+.10))
            jaw=arms[side].get('jaw')
            if jaw is None:
                return fail('Need measured jaw to distinguish pickup from possible held-object recovery.')
            if .018<jaw<.8:
                # Possible held object: never open, descend to source, or use its
                # projected silhouette to infer the gripper's world yaw.
                s.update(held=True,recovery=True,orientation_unknown=True)
                start_scan()
            elif jaw>=.8:
                a=(float(o['source_yaw'])+math.pi)%math.pi-math.pi/2
                turn=(float(o['turn_rad'])+math.pi)%(2*math.pi)-math.pi
                src=list(o['source'])
                s.update(stage='open',src=src,yaw=a,goal=a+turn,recovery=False,
                         height=max(s['height'],src[2]+.10))
            else:
                return fail('Nearly shut jaw with no history: inspect for an empty or lost grasp; no blind reopening.')
        else:
            return fail('Unsupported kind.')

    if s.get('halted'):
        return result('fail_reobserve',reason=s['halted'])
    if (s['kind'],s['arm'],s.get('digit'))!=(o['kind'],side,o.get('digit')):
        return fail('State identity mismatch; do not reuse a held-object state for another transfer.')
    if o['kind']=='transfer' and dist(s['dst'],o['dest'])>.002:
        return fail('Destination changed: explicitly reconcile pending route before continuing.')
    if int(o.get('remaining_steps',0))<25:
        return fail('Low remaining motion budget; retain grip and request recovery.')

    p=s.get('pending')
    if p:
        m=arms[p['arm']]
        status=str(last.get('status') or '').lower()
        reason=str(last.get('reason') or '').lower()
        complete=status in ('completed','success','succeeded','ok')
        if last.get('op') not in (None,p['op']):
            return fail('Receipt does not match pending operation.')
        if p['op']=='pose_preview':
            q=last.get('q_rad')
            if q is not None and len(q)==6 and all(math.isfinite(v) for v in q) and not any(w in status for w in ('fail','error','reject')):
                ref=s['scan_q'] if s['stage'].startswith('scan_') else m['q']
                cost=max(abs(x-y) for x,y in zip(q,ref))
                s['candidates'].append([cost,p['yaw']])
            s['scan_i']+=1
            s.pop('pending',None)
        else:
            if p['op']=='move_tcp':
                error=dist(m['tcp'],p['xyz'])
                s['last_tcp_error']=error
                tolerance=.004 if s['stage'] in ('down','place') else .008
                ok=complete and error<=tolerance
            elif p['op']=='grip':
                jaw=m.get('jaw')
                if jaw is None:
                    return fail('Missing jaw measurement after grip.')
                ok=complete and (jaw>=.85 if p['jaw']==1 else .018<jaw<.8)
                if complete and p['jaw']==0 and jaw<=.018:
                    return fail('Likely empty close: reobserve grasp center; do not lift or transit.')
            else:
                ok=max(abs(x-y) for x,y in zip(m['q'],p['q']))<=.025
                ok=ok and (complete or reason in ('execution_timeout','tracking_error'))
            if not ok:
                s['tries']+=1
                if (complete or reason in ('execution_timeout','tracking_error','tracking error')) and s['tries']<=2:
                    return dict(p,state=s)
                return fail('Action did not converge: '+reason+'. Inspect measured TCP error and grip; infeasible paths need new geometry, not repeated commands.')
            s.pop('pending',None)
            s['tries']=0
            st=s['stage']
            if p['op']=='grip':
                s['held']=p['jaw']==0
            if st=='rotate':
                # Preserve selected yaw representation in subsequent commands.
                offset=p['yaw']-s['next_yaw']
                s['goal']+=offset
                s['yaw']=p['yaw']
                s['stage']='rotation_check'
            elif st=='home_joint':
                pass
            else:
                transitions={'open':'clear','clear':'above','above':'down','down':'close',
                             'close':'lift','lift':'carry','carry':'rotation_check',
                             'recover_lift':'recover_carry','recover_carry':'recover_observe',
                             'place':'release','release':'retreat','retreat':'park','park':'done',
                             'home_clear':'home_park','home_park':'home_joint','home_open':'home_next'}
                if st not in transitions:
                    return fail('Unexpected pending-action stage.')
                s['stage']=transitions[st]

    if o['kind']=='home':
        if s['stage']=='home_next':
            s['index']+=1
            s['stage']='home_clear'
        if s['index']>=2:
            if all(max(abs(x-y) for x,y in zip(arms[k]['q'],arms[k]['home_q']))<=.025 and arms[k].get('jaw',0)>=.85 for k in ('left','right')):
                return result('done')
            return fail('Final home verification failed.')
        k=('left','right')[s['index']]
        m=arms[k]; h=max(tz+.15,m['home_tcp'][2])
        if s['stage']=='home_clear':
            return issue('move_tcp',k,xyz=[m['tcp'][0],m['tcp'][1],max(h,m['tcp'][2])],yaw=0.)
        if s['stage']=='home_park':
            return issue('move_tcp',k,xyz=[m['home_tcp'][0],m['home_tcp'][1],h],yaw=0.)
        if s['stage']=='home_joint':
            d=max(abs(x-y) for x,y in zip(m['q'],m['home_q']))
            if d>.025:
                other=arms['right' if k=='left' else 'left']
                if m['tcp'][2]<tz+.09 or dist(m['tcp'],other['tcp'])<.18:
                    return fail('Unsafe clearance for incremental joint homing; request observed collision-aware separation.')
                f=min(1.,.10/d)
                return issue('joint',k,q=[x+f*(y-x) for x,y in zip(m['q'],m['home_q'])],steps=10)
            s['stage']='home_open'
        return issue('grip',k,jaw=1)

    if s['held'] and not (.018<arms[side].get('jaw',0)<.8):
        return fail('Possible grip loss/change; inspect object before further motion. Do not open.')
    st=s['stage']; h=s['height']; dst=s['dst']
    if st.startswith('scan_'):
        if dist(arms[side]['tcp'],s['scan_xyz'])>.003 or max(abs(x-y) for x,y in zip(arms[side]['q'],s['scan_q']))>.03:
            return fail('Arm changed during stationary pose queries; cannot estimate current yaw reliably.')
        i=s['scan_i']
        if i<len(s['angles']):
            return issue('pose_preview',side,xyz=s['scan_xyz'],yaw=s['angles'][i])
        if not s['candidates']:
            return fail('No usable public pose candidates; need measured/calibrated gripper orientation or retained grasp history.')
        cost,y=min(s['candidates'])
        if st=='scan_coarse':
            start_scan(y)
            return issue('pose_preview',side,xyz=s['scan_xyz'],yaw=s['angles'][0])
        if cost>.10:
            return fail('Preview poses do not closely match measured absolute joints; current top-down yaw remains unknown. Retain grip.')
        s.update(yaw=y,stage='recover_lift',yaw_fit_error=cost)
        st='recover_lift'
    if st=='recover_observe':
        return fail('Held object moved to destination clearance without release. Obtain a reliable upright correction and verify center/contact. Zero turn from an occluded view was not treated as upright confirmation; preserve this state.')
    if st=='done':
        return result('done')
    if st=='rotation_check':
        delta=s['goal']-s['yaw']
        if abs(delta)<.001:
            s['stage']='place'; st='place'
        else:
            s.update(stage='rotation_preview',next_yaw=s['yaw']+max(-.15,min(.15,delta)),scan_i=0,candidates=[])
            st='rotation_preview'
    if st=='rotation_preview':
        i=s['scan_i']
        if i<3:
            return issue('pose_preview',side,xyz=[dst[0],dst[1],h],yaw=s['next_yaw']+(0,2*math.pi,-2*math.pi)[i])
        if not s['candidates']:
            return fail('No usable rotation preview; retain grip and revise high waypoint.')
        cost,y=min(s['candidates'])
        if cost>.75:
            return fail('Rotation requires a large absolute joint change; revise posture at clearance.')
        s['stage']='rotate'
        return issue('move_tcp',side,xyz=[dst[0],dst[1],h],yaw=y)
    if st in ('open','close','release'):
        return issue('grip',side,jaw=0 if st=='close' else 1)
    if st in ('clear','recover_lift'):
        t=arms[side]['tcp']; xyz=[t[0],t[1],max(h,t[2])]
    elif st in ('above','lift'):
        xyz=[s['src'][0],s['src'][1],h]
    elif st=='down':
        xyz=s['src']
    elif st in ('carry','recover_carry','retreat'):
        xyz=[dst[0],dst[1],h]
    elif st=='place':
        xyz=dst
    elif st=='park':
        t=arms[side]['home_tcp']; xyz=[t[0],t[1],max(h,t[2])]
    else:
        return fail('Unknown controller stage.')
    return issue('move_tcp',side,xyz=xyz,yaw=s['yaw'])
