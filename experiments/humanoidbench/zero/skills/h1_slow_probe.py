# Requires skills/h1_stance.py first.
# Experimental slow motion only; NOT validated for a full walking episode.
# Pitch drifts forward. Do not extrapolate short-horizon survival to success.
def h1_slow_probe_init(observation):
    return {'balance': h1_stance_init(observation),
            'origin': observation['state']['qpos'][0],
            'start': observation['additional_info']['seconds']}

def h1_slow_probe_action(controller, observation):
    t = observation['additional_info']['seconds'] - controller['start']
    tau = max(0, t-0.6)
    balance = controller['balance']
    balance['reference_x'] = controller['origin']+0.025*tau
    q = h1_stance_joints(balance, observation)
    phase = 2*math.pi*tau/0.6
    ramp = min(1.0, tau/1.0)
    for h, offset in [(2, 0), (7, math.pi)]:
        c = math.cos(phase+offset)
        lift = 0.12*max(0, math.sin(phase+offset))*ramp
        q[h] += 0.045*c*ramp-0.5*lift
        q[h+1] += lift
        q[h+2] += -0.045*c*ramp-0.5*lift
    return np.clip(2*(q-balance['low'])/(balance['high']-balance['low'])-1, -1, 1).tolist()
