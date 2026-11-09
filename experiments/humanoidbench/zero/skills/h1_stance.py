import math
import numpy as np

# Experimental stationary component, NOT a walking policy.
# Initialize from a fresh public observation. Stop on terminated/truncated.
# Functions and dictionaries are used because this host disallows classes.
def h1_stance_init(observation):
    channels = observation['action']['channels']
    nominal = np.array([observation['state']['qpos'][c['qpos']] for c in channels])
    for h in [2, 7]:
        nominal[h] = -0.3
        nominal[h+1] = 0.6
        nominal[h+2] = -0.3
    return {'low': np.array([c['min_rad'] for c in channels]),
            'high': np.array([c['max_rad'] for c in channels]),
            'nominal': nominal,
            'reference_x': observation['state']['qpos'][0]}

def h1_stance_joints(controller, observation):
    p = observation['state']['qpos']
    v = observation['state']['qvel']
    w, x, y, z = p[3:7]
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(max(-1, min(1, 2*(w*y-z*x))))
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    q = controller['nominal'].copy()
    for h in [2, 7]:
        q[h] += np.clip(pitch+0.2*v[4], -0.5, 0.5)
        q[h+2] += np.clip(8*(p[0]-controller['reference_x'])+4*v[0]-0.03, -0.55, 0.8)
        q[h-1] = np.clip(roll+0.2*v[3]-0.25*v[1], -0.25, 0.25)
        q[h-2] = np.clip(0.4*yaw+0.08*v[5], -0.2, 0.2)
    return q

def h1_stance_action(controller, observation):
    q = h1_stance_joints(controller, observation)
    return np.clip(2*(q-controller['low'])/(controller['high']-controller['low'])-1, -1, 1).tolist()
