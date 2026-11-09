import math

def decide(o):
    """One measured, persistent robot decision; standard library only."""
    s = dict(o.get('state') or {})
    arms = o['arms']
    arm = o['arm']
    table = float(o['table_z'])
    last = o.get('last') or {}

    def dist(a, b):
        return math.sqrt(sum((float(x)-float(y))**2 for x, y in zip(a, b)))

    def emit(op, side=None, **kw):
        r = dict(op=op, state=s, **kw)
        if side is not None:
            r['arm'] = side
        return r

    def fail(reason):
        s['halted'] = reason
        return emit('fail_reobserve', reason=reason)

    def move(side, xyz, yaw):
        xyz = [float(v) for v in xyz]
        if not all(math.isfinite(v) for v in xyz+[yaw]):
            return fail('Nonfinite target; refresh calibrated targets.')
        if not (abs(xyz[0]) <= .62 and -.45 <= xyz[1] <= .52 and table-.005 <= xyz[2] <= 1.17):
            return fail('Target outside TCP bounds; revise geometry without releasing any held object.')
        return emit('move_tcp', side, xyz=xyz, yaw=float(yaw))

    if s.get('halted'):
        return emit('fail_reobserve', reason=s['halted'])
    if int(o.get('remaining_steps', 0)) < 25:
        return fail('Low step reserve: inspect current grip and pose; no automatic release.')

    signature = [o['kind'], arm, o.get('digit'), o.get('source'), o.get('dest')]
    if s and s.get('signature') != signature:
        return fail('State belongs to another operation; reconcile held object before resetting state.')
    if not s:
        s.update(signature=signature, phase=0, tries=0, held=False)
        if o['kind'] == 'transfer':
            src, dst = o['source'], o['dest']
            if len(src) != 3 or len(dst) != 3:
                return fail('Need calibrated source and destination grasp centers.')
            # A silhouette axis is unsigned: choose one equivalent closing axis,
            # then preserve its signed turn rather than independently wrapping it.
            a = float(o['source_yaw']) + math.pi/2
            a = (a + math.pi/2) % math.pi - math.pi/2
            turn = float(o['turn_rad'])
            turn = (turn + math.pi) % (2*math.pi) - math.pi
            h = max(table+.14, float(src[2])+.10, float(dst[2])+.10)
            if h > 1.14:
                return fail('Insufficient overhead clearance; revise approach.')
            s.update(src=list(src), dst=list(dst), yaw=a, final_yaw=a+turn, height=h)
            if float(arms[arm].get('jaw', 0)) < .8:
                return fail('Initial active jaw is not open; inspect for a previously held object before opening.')
        elif o['kind'] == 'home':
            # Homing must not silently discard an object.
            if any(float(arms[k].get('jaw', 0)) < .8 for k in ('left', 'right')):
                return fail('Before homing, verify both grippers empty and open; refusing to drop an unknown object.')
        else:
            return fail('Unknown operation kind.')

    # Receipts are checked together with fresh proprio, never in isolation.
    p = s.get('pending')
    if p:
        side = p['arm']
        measured = arms[side]
        status = str(last.get('status', '')).lower()
        reason = str(last.get('reason', '')).lower()
        bad = any(w in status for w in ('fail', 'error', 'reject', 'block', 'abort'))
        bad = bad or any(w in reason for w in ('collision', 'unreachable', 'tracking error', 'tracking_error'))
        if last.get('op') not in (None, p['op']):
            return fail('Receipt does not match pending action; reconcile executor state.')
        if p['op'] == 'move_tcp':
            error = dist(measured['tcp'], p['xyz'])
            # Position alone cannot establish successful wrist rotation.
            ok = error <= p.get('tol', .008) and not bad
        elif p['op'] == 'grip':
            jaw = measured.get('jaw')
            if jaw is None:
                return fail('Missing measured jaw after grip; inspect grasp before continuing.')
            ok = (jaw >= .85 if p['jaw'] == 1 else jaw <= .55) and not bad
            if p['jaw'] == 0 and not bad and jaw <= .018:
                s['held'] = False
                return fail('Close reached nearly shut: likely empty grasp. Reobserve object center/contact before retrying; do not transit.')
        else:
            ok = max(abs(x-y) for x, y in zip(measured['q'], p['q'])) <= .045 and not bad
        if not ok:
            s['tries'] += 1
            if s['tries'] >= 3:
                return fail('Action failed to converge. Inspect fresh TCP/jaw and obstruction; if holding, keep closed. For crossover, request an observed staged handoff or revised waypoint.')
            # Tracking errors near the target can be retried, including rotation;
            # collision/reach failures require new geometry, not blind repetition.
            if any(w in reason+' '+status for w in ('collision', 'unreachable', 'blocked')):
                return fail('Planner obstruction/reach failure: keep current grip; inspect inactive arm and revise path or stage a handoff.')
            retry = {k:v for k,v in p.items() if k != 'tol'}
            return dict(retry, state=s)
        s.pop('pending', None)
        s['tries'] = 0
        if p['op'] == 'grip':
            s['held'] = p['jaw'] == 0
        # Homing joint increments are recomputed from measured joints.
        if not p.get('op') == 'joint':
            s['phase'] += 1

    def queue(r, tol=None):
        if r['op'] == 'fail_reobserve':
            return r
        s['pending'] = {k:v for k,v in r.items() if k != 'state'}
        if tol is not None:
            s['pending']['tol'] = tol
        return r

    if o['kind'] == 'home':
        phase = s['phase']
        if phase < 2:
            side = ('left', 'right')[phase]
            t = arms[side]['tcp']
            return queue(move(side, [t[0], t[1], max(t[2], table+.15)], 0))
        side = ('left', 'right')[0 if phase < 4 else 1]
        m = arms[side]
        if phase in (2, 4):
            q, goal = m['q'], m['home_q']
            if max(abs(x-y) for x,y in zip(q, goal)) > .025:
                if m['tcp'][2] < table+.09:
                    return fail('Direct homing posture became low; reobserve and plan a collision-aware clearance move.')
                other = arms['right' if side == 'left' else 'left']['tcp']
                if dist(m['tcp'], other) < .18:
                    return fail('Arms too close for unplanned joint homing; separate with observed collision-aware waypoints.')
                delta = max(abs(x-y) for x,y in zip(q, goal))
                scale = min(1., .10/delta)
                target = [x+scale*(y-x) for x,y in zip(q, goal)]
                return queue(emit('joint', side, q=target, steps=10))
            s['phase'] += 1
            phase = s['phase']
        if phase in (3, 5):
            return queue(emit('grip', side, jaw=1))
        if phase >= 6:
            for k in ('left', 'right'):
                m = arms[k]
                if max(abs(x-y) for x,y in zip(m['q'],m['home_q'])) > .025 or m.get('jaw',0) < .85:
                    return fail('Final home verification failed; both measured home joints and open jaws required.')
            return emit('done')

    src, dst = s['src'], s['dst']
    h, a, b = s['height'], s['yaw'], s['final_yaw']
    phase = s['phase']
    if s['held'] and float(arms[arm].get('jaw', 0)) <= .018:
        return fail('Jaw collapsed during transport: possible lost object. Reobserve before further motion.')
    if phase == 0:
        return queue(emit('grip', arm, jaw=1))
    if phase == 1:
        t = arms[arm]['tcp']
        return queue(move(arm, [t[0],t[1],max(h,t[2])], a))
    if phase == 2:
        return queue(move(arm, [src[0],src[1],h], a))
    if phase == 3:
        return queue(move(arm, src, a), .004)
    if phase == 4:
        return queue(emit('grip', arm, jaw=0))
    if phase == 5:
        return queue(move(arm, [src[0],src[1],h], a))
    if phase == 6:
        return queue(move(arm, [src[0],src[1],h], b))
    if phase == 7:
        return queue(move(arm, [dst[0],dst[1],h], b))
    if phase == 8:
        return queue(move(arm, dst, b), .004)
    if phase == 9:
        # Release is reachable only after a verified placement descent.
        return queue(emit('grip', arm, jaw=1))
    if phase == 10:
        return queue(move(arm, [dst[0],dst[1],h], b))
    if phase == 11:
        home = arms[arm]['home_tcp']
        return queue(move(arm, [home[0],home[1],max(h,home[2])], b))
    return emit('done')
