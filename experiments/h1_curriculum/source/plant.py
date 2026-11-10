"""Fixed-plant H1 command tracking: evaluate one frozen numeric checkpoint per suite."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

ENV_ID = "h1-walk-v0"
DT = 0.02
NQ, NV, NU = 26, 25, 19
HERE = Path(__file__).resolve().parent
FEATURE_NAMES = tuple(f"qpos_{i}" for i in range(2,NQ)) + tuple(f"qvel_{i}" for i in range(NV)) + (
    "target_vx", "root_roll", "root_pitch", "root_yaw", "left_contact", "right_contact",
    "left_foot_x", "left_foot_y", "left_foot_z", "right_foot_x", "right_foot_y", "right_foot_z",
)
FOOT_NAMES = ("left_ankle_link", "right_ankle_link")
STAND_SPEED_RMS = .12
STAND_EXCURSION = .25
MIN_ROOT = .63
UPRIGHT_COS = .70
WALK_SPEED = .50
WALK_ERROR = .17
WALK_RMSE = .23

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False)+"\n")

def native_env():
    import gymnasium as gym
    import humanoid_bench  # registers environments

    env = gym.make(ENV_ID, render_mode=None)
    u = env.unwrapped
    if (u.model.nq,u.model.nv,u.model.nu) != (NQ,NV,NU) or not np.isclose(u.dt,DT):
        env.close()
        raise RuntimeError("Native H1 dimensions or tick differ from registered contract")
    return env

def channel_info(env):
    import mujoco

    u = env.unwrapped
    u.reset(seed=0)
    out = []
    for k,tr in enumerate(u.model.actuator_trnid[:,0]):
        tr = int(tr)
        q = int(u.model.jnt_qposadr[tr]); v = int(u.model.jnt_dofadr[tr])
        lo = float(u.action_low[k]); hi = float(u.action_high[k])
        ref = float(u.model.key_qpos[u.keyframe,q])
        out.append(dict(joint=mujoco.mj_id2name(u.model,mujoco.mjtObj.mjOBJ_JOINT,tr),
                        qpos=q,qvel=v,min_rad=lo,max_rad=hi,
                        midpoint_rad=(lo+hi)/2,native_keyframe_rad=ref,
                        keyframe_normalized_action=float(np.clip(2*(ref-lo)/(hi-lo)-1,-1,1))))
    return out

def describe():
    env = native_env()
    try:
        channels=channel_info(env)
        u=env.unwrapped
        return dict(
            task=ENV_ID,dt_s=DT,format="h1-curriculum-v1",
            feature_names=FEATURE_NAMES,action_names=[r["joint"] for r in channels],action_channels=channels,
            native_keyframe_qpos=u.model.key_qpos[u.keyframe].tolist(),
            feature_meanings={
                "qpos_2":"Pelvis world height above floor, meters.",
                "qpos_3..6":"Pelvis orientation quaternion w,x,y,z.",
                "qpos_7..25":"Native joint angles in radians; see qpos index by named action channel.",
                "qvel_0..2":"Pelvis world linear velocity in X,Y,Z, m/s. Desired travel is world +X.",
                "qvel_3..5":"Pelvis local angular velocity, rad/s.",
                "qvel_6..24":"Native joint angular velocities in rad/s; see qvel index by channel.",
                "target_vx":"Continuous desired world-X velocity, m/s; only command input.",
                "root_roll/root_pitch/root_yaw":"Euler angles derived from the pelvis quaternion, radians.",
                "left_contact/right_contact":"1 when that foot bears >5 N of normal floor contact force, else 0.",
                "*_foot_x/y":"Ankle body origin minus pelvis, expressed in world x/y, meters.",
                "*_foot_z":"Ankle body origin height above floor, meters.",
            },
            action_mapping="Native normalized joint position target a∈[-1,1] maps to q=lo+(a+1)(hi-lo)/2. Zero means each actuator midpoint, not neutral/hold. Native servos are unchanged.",
            reset="One native reset per case plus deterministic independent pelvis tilt jitter (roll and pitch each <=3deg). No clock, absolute horizontal position, case ID or phase provided to policy.",
            shape=[len(FEATURE_NAMES),NU],
        )
    finally:
        env.close()

def rotation(q):
    w,x,y,z=np.asarray(q,dtype=np.float64)/np.linalg.norm(q)
    return (
        math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
        math.asin(float(np.clip(2*(w*y-z*x),-1,1))),
        math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)),
    )

def reset_pose(env,seed):
    import mujoco

    env.reset(seed=seed)
    u=env.unwrapped
    rng=np.random.default_rng(np.random.SeedSequence([seed,51931]))
    roll,pitch=rng.uniform(-math.pi/60,math.pi/60,2)
    cr,sr=math.cos(roll/2),math.sin(roll/2)
    cp,sp=math.cos(pitch/2),math.sin(pitch/2)
    tilt=np.array([cr*cp,sr*cp,cr*sp,-sr*sp])
    q=u.data.qpos.copy()
    mujoco.mju_mulQuat(q[3:7],tilt,u.data.qpos[3:7].copy())
    q[3:7]/=np.linalg.norm(q[3:7])
    u.set_state(q,u.data.qvel.copy())
    return [float(roll),float(pitch)]

class Sensors:
    def __init__(self,u):
        import mujoco

        self.u=u
        self.floor=mujoco.mj_name2id(u.model,mujoco.mjtObj.mjOBJ_GEOM,"floor")
        self.feet=tuple(mujoco.mj_name2id(u.model,mujoco.mjtObj.mjOBJ_BODY,n) for n in FOOT_NAMES)
        self.torso=mujoco.mj_name2id(u.model,mujoco.mjtObj.mjOBJ_BODY,"torso_link")
        if min(self.floor,self.torso,*self.feet)<0:
            raise RuntimeError("Missing native floor/ankle/torso identifiers")
        self.foot_idx={k:i for i,k in enumerate(self.feet)}

    def sample(self,cmd):
        import mujoco

        u=self.u; d=u.data
        foot_normal=np.zeros(2); nonfoot=0.
        wrench=np.zeros(6)
        for i in range(d.ncon):
            c=d.contact[i]
            if c.efc_address<0: continue
            if c.geom1==self.floor: geom=c.geom2
            elif c.geom2==self.floor: geom=c.geom1
            else: continue
            mujoco.mj_contactForce(u.model,d,i,wrench)
            idx=self.foot_idx.get(int(u.model.geom_bodyid[geom]))
            if idx is None: nonfoot+=wrench[0]
            else: foot_normal[idx]+=wrench[0]
        foot_abs=d.xpos[np.asarray(self.feet)].copy()
        foot_rel=foot_abs.copy()
        foot_rel[:,:2]-=d.qpos[None,:2]
        contacts=foot_normal>5
        angles=rotation(d.qpos[3:7])
        features=np.r_[d.qpos[2:],d.qvel,cmd,angles,contacts.astype(float),foot_rel.reshape(-1)]
        sample=dict(qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),target_vx=float(cmd),
                    root_rpy=list(angles),foot_contact=contacts.tolist(),foot_xyz_m=foot_abs.tolist(),
                    foot_normal_n=foot_normal.tolist(),nonfoot_support_n=float(nonfoot),
                    torso_upright_cos=float(d.xmat[self.torso,8]))
        return features,sample

def segments_for(c):
    if c["kind"] in ("stand","walk"):
        return [(0,1000,0. if c["kind"]=="stand" else WALK_SPEED)]
    segments=[]; i=0
    for dur,goal in c["segments"]:
        n=round(dur/DT)
        segments.append((i,i+n,float(goal))); i+=n
    if i!=1000:
        raise ValueError("Transition schedule must run 1000 ticks")
    return segments

def schedule(segments,ramp_s):
    goals=np.zeros(segments[-1][1]); prev=0.
    ramp=round(ramp_s/DT)
    for a,b,g in segments:
        goals[a:b]=g
        if ramp and a:
            k=min(b-a,ramp)
            goals[a:a+k]=prev+(g-prev)*(np.arange(k)+1)/k
        prev=g
    return goals

def longest_run(xs):
    prev=None; run=best=0
    for x in xs:
        run=run+1 if x!=prev else 1
        best=max(best,run); prev=x
    return best

def find_touchdowns(trace):
    """A step requires ≥80 ms flight, clearance ≥1.2 cm and swing advance ≥3.5 cm."""
    contacts=np.asarray([r["foot_contact"] for r in trace],dtype=bool)
    feet=np.asarray([r["foot_xyz_m"] for r in trace])
    events=[]
    for side in range(2):
        off=None
        for i in range(1,len(trace)):
            if contacts[i-1,side] and not contacts[i,side]:
                off=i
            elif not contacts[i-1,side] and contacts[i,side]:
                if off is not None and i-off>=4 and contacts[i:i+3,side].sum()>=2:
                    previous=feet[off-1,side]
                    advance=feet[i,side,0]-previous[0]
                    clearance=feet[off:i,side,2].max()-previous[2]
                    if advance>=.035 and clearance>=.012:
                        events.append(dict(step=i,side=("left","right")[side],advance_m=float(advance),
                                           clearance_m=float(clearance),air_s=(i-off)*DT))
                off=None
    return sorted(events,key=lambda r:(r["step"],r["side"]))

def consecutive(mask, n):
    count=0
    for hit in mask:
        count=count+1 if hit else 0
        if count>=n: return True
    return False

def upright_prefix(trace):
    """Last continuously upright prefix, before sustained physical loss of balance."""
    streak=[0,0,0]
    for i,row in enumerate(trace):
        failed=(row["qpos"][2]<MIN_ROOT,row["torso_upright_cos"]<UPRIGHT_COS,
                row["nonfoot_support_n"]>8.)
        streak=[count+1 if bad else 0 for count,bad in zip(streak,failed)]
        if max(streak)>=6:
            return max(0,i-5)
    return len(trace)-1

def score(c,segments,goals,trace,reason,events):
    q=np.asarray([r["qpos"] for r in trace]); v=np.asarray([r["qvel"] for r in trace])
    all_streak=longest_run([e["side"] for e in events])
    upright=np.asarray([r["torso_upright_cos"] for r in trace])
    nonfoot=np.asarray([r["nonfoot_support_n"] for r in trace])
    physically_down=(consecutive(q[:,2]<MIN_ROOT,6) or consecutive(upright<UPRIGHT_COS,6)
                     or consecutive(nonfoot>8.,6) or q[-1,2]<MIN_ROOT or upright[-1]<UPRIGHT_COS)
    n=len(trace)-1
    up=upright_prefix(trace)
    complete=n==len(goals) and reason=="horizon"
    scores=[]
    for a,b,g in segments:
        if n<b:
            scores.append(dict(start_s=a*DT,end_s=b*DT,target_vx_mps=g,passed=False,reason="incomplete"))
            continue
        warm=0 if c["kind"]=="stand" else (150 if c["kind"]=="walk" else (80 if g==0 else 65))
        i=a+warm
        duration=(b-i)*DT
        delta=q[b,:2]-q[i,:2]
        vel=v[i+1:b+1,:2]
        fwd=float(delta[0]/duration)
        excursion=float(np.linalg.norm(q[i:b+1,:2]-q[i,:2],axis=1).max())
        horizontal_rms=float(np.sqrt((vel**2).sum(1).mean()))
        mae_y=float(np.mean(np.abs(vel[:,1])))
        err=float(np.sqrt(np.mean((vel[:,0]-goals[i:b])**2)))
        e=[r["side"] for r in events if i<r["step"]<=b]
        alt=longest_run(e)
        alt_total=sum(x!=y for x,y in zip(e,e[1:]))
        if g==0:
            checks=dict(excursion=excursion<=(STAND_EXCURSION if c["kind"]=="stand" else .20),
                        speed=horizontal_rms<=STAND_SPEED_RMS)
        elif c["kind"]=="walk":
            checks=dict(progress=fwd>=WALK_SPEED-WALK_ERROR,mean_speed=abs(fwd-g)<=WALK_ERROR,
                        speed_rmse=err<=WALK_RMSE,lateral=mae_y<=.18,
                        alternating=alt>=6 and e.count("left")>=3 and e.count("right")>=3)
        else:
            limit=min(.15,.07+.15*g)
            checks=dict(progress=fwd>=max(.08,g-limit),mean_speed=abs(fwd-g)<=limit,
                        speed_rmse=err<=.22,lateral=mae_y<=.18,alternating=alt>=2)
        scores.append(dict(start_s=a*DT,end_s=b*DT,target_vx_mps=g,warmup_s=warm*DT,
            x_m=float(delta[0]),y_m=float(delta[1]),mean_vx_mps=fwd,max_xy_excursion_m=excursion,
            horizontal_speed_rms_mps=horizontal_rms,vx_rmse_mps=err,mean_abs_vy_mps=mae_y,
            left_touchdowns=e.count("left"),right_touchdowns=e.count("right"),
            alternating_touchdowns=alt_total,longest_alternating_run=alt,
            passed=all(checks.values()),checks=checks))
    transitions=0
    for i,e in enumerate(events[1:],1):
        if events[i-1]["side"]!=e["side"]:
            transitions=e["step"]; break
    return dict(passed=bool(complete and not physically_down and all(s["passed"] for s in scores)),
         steps=n,elapsed_s=n*DT,termination=reason,physically_down=bool(physically_down),
         upright_prefix_s=up*DT,upright_prefix_x_m=float(q[up,0]-q[0,0]),
         x_m=float(q[-1,0]-q[0,0]),y_m=float(q[-1,1]-q[0,1]),
         min_root_z_m=float(q[:,2].min()),final_root_z_m=float(q[-1,2]),
         min_torso_upright_cos=float(upright.min()),max_nonfoot_support_n=float(nonfoot.max()),
         mean_vx_mps=float((q[-1,0]-q[0,0])/(max(1,n)*DT)),
         alternating_touchdowns=sum(x["side"]!=y["side"] for x,y in zip(events,events[1:])),
         longest_alternating_run=all_streak,left_touchdowns=sum(r["side"]=="left" for r in events),
         right_touchdowns=sum(r["side"]=="right" for r in events),
         first_touchdown_s=events[0]["step"]*DT if events else None,
         first_alternation_s=transitions*DT if transitions else None,segments=scores)

def diagnostic(trace, channels, events):
    def pick(r):
        q=r["qpos"]; v=r["qvel"]; action=r["action"]
        # State samples leave the full body trace in the per-case file.
        return {"t_s":r["t_s"],"target_vx_mps":r["target_vx"],"pelvis_xyz_m":[round(x,4) for x in q[:3]],
            "pelvis_vxyz_mps":[round(x,4) for x in v[:3]],"pelvis_rpy_rad":[round(x,4) for x in r["root_rpy"]],
            "foot_contact":r["foot_contact"],"foot_normal_n":[round(x,2) for x in r["foot_normal_n"]],
            "foot_relative_xy_z_m":[[round(x-q[0],4),round(y-q[1],4),round(z,4)] for x,y,z in r["foot_xyz_m"]],
            "leg_joints_rad":{k["joint"]:round(q[k["qpos"]],4) for k in channels if any(y in k["joint"] for y in ("hip","knee","ankle"))},
            "native_leg_targets":{k["joint"]:round(action[i],4) for i,k in enumerate(channels) if action is not None and any(y in k["joint"] for y in ("hip","knee","ankle"))}}
    stop=len(trace)-1
    indices={0,5,12,25,38,50,75,100,150,200,300,500,750,1000,max(0,stop-10),stop}
    indices.update(e["step"] for e in events[:4])
    return dict(samples=[pick(trace[i]) for i in sorted(indices) if i<=stop],
                first_touchdowns=events[:16],
                first_nonfoot_support_s=next((r["t_s"] for r in trace if r["nonfoot_support_n"]>8),None),
                initial_root_state=trace[0]["qpos"][:7])

def rollout(env,policy,c,channels):
    segments=segments_for(c)
    goals=schedule(segments,float(c.get("ramp_s",0)))
    tilt=reset_pose(env,c["seed"])
    u=env.unwrapped; sensors=Sensors(u)
    obs,row=sensors.sample(goals[0])
    trace=[dict(step=0,t_s=0.,native_reward=None,action=None,**row)]
    reason="horizon"
    for i,cmd in enumerate(goals):
        # Update command for this action, but do not expose time, task labels or evaluator state.
        obs[len(range(2,NQ))+NV]=cmd
        a=np.asarray(policy.act(obs),dtype=np.float64)
        if a.shape!=(NU,) or not np.isfinite(a).all() or np.any(np.abs(a)>1):
            raise ValueError("Numeric checkpoint did not produce 19 finite normalized targets")
        _,reward,term,truncated,_=env.step(a)
        obs,row=sensors.sample(cmd)
        if not np.isfinite(obs).all() or not math.isfinite(reward):
            reason="nonfinite_native_transition"
            break
        trace.append(dict(step=i+1,t_s=(i+1)*DT,native_reward=float(reward),action=a.tolist(),**row))
        if term:
            reason="native_fall"
            break
        if truncated and i+1!=len(goals):
            reason="native_early_truncation"
            break
    touchdown=find_touchdowns(trace)
    metrics=score(c,segments,goals,trace,reason,touchdown)
    metrics["diagnostic"]=diagnostic(trace,channels,touchdown)
    metrics["reset_tilt_rad"]=tilt
    metrics["native_walk_return"]=sum(r["native_reward"] for r in trace[1:])
    return metrics,trace

def render(trace,path):
    import imageio.v2 as iio
    import mujoco
    from PIL import Image,ImageDraw,ImageFont

    env=native_env()
    u=env.unwrapped; m=u.model; d=mujoco.MjData(m)
    floor=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"floor")
    max_x=max(abs(r["qpos"][0]) for r in trace)
    m.geom_size[floor,:2]=np.maximum(m.geom_size[floor,:2],[max_x+6,6])
    m.vis.global_.offwidth=640; m.vis.global_.offheight=480
    camera=mujoco.MjvCamera()
    camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth=140; camera.elevation=-18; camera.distance=3.7
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font=ImageFont.truetype(font_path,17) if Path(font_path).is_file() else ImageFont.load_default()
    try:
        with mujoco.Renderer(m,height=480,width=640) as renderer, iio.get_writer(str(path),fps=25,codec="libx264") as w:
            for r in trace[::2]+([trace[-1]] if len(trace)%2==0 else []):
                d.qpos[:]=r["qpos"]; d.qvel[:]=r["qvel"]
                mujoco.mj_forward(m,d)
                camera.lookat[:]=(d.qpos[0]+.25,d.qpos[1],.90)
                renderer.update_scene(d,camera=camera)
                frame=Image.fromarray(renderer.render())
                overlay=ImageDraw.Draw(frame)
                overlay.rectangle((0,0,640,43), fill=(18,23,31))
                overlay.text((12,11), f"t {r['t_s']:5.2f}s   Target {r['target_vx']:.2f} m/s   Actual {r['qvel'][0]:+.2f} m/s",
                             fill=(244,246,248),font=font)
                w.append_data(np.asarray(frame))
    finally:
        env.close()

def evaluate(args):
    from weight_policy import Policy

    manifest=json.loads((HERE/"suites.json").read_text())
    cases=manifest["suites"][args.suite]
    out=Path(args.out)
    out.mkdir(parents=True)
    t=time.monotonic()
    env=native_env()
    try:
        channels=channel_info(env)
        policy=Policy.load(Path(args.weights),FEATURE_NAMES,tuple(k["joint"] for k in channels))
        rows=[]
        for case in cases:
            ident=f'{case["kind"]}-{case["seed"]}'
            results,trace=rollout(env,policy,case,channels)
            trace_name=ident+".trace.json"
            write(out/trace_name,trace)
            row=dict(case_id=ident,seed=case["seed"],kind=case["kind"],trace=trace_name,**results)
            if args.video:
                video_name=ident+".mp4"
                render(trace,out/video_name)
                row["video"]=video_name
            rows.append(row)
        good=sum(r["passed"] for r in rows)
        summary=dict(suite=args.suite,passed=good==len(rows),passed_cases=good,case_count=len(rows),
                     elapsed_wall_s=time.monotonic()-t,cases=rows)
        write(out/"summary.json",summary)
        print(json.dumps({k:v for k,v in summary.items() if k!="cases"},ensure_ascii=True),flush=True)
    finally:
        env.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    commands.add_parser("describe")
    a=commands.add_parser("evaluate")
    a.add_argument("--weights",required=True)
    a.add_argument("--suite",required=True,choices=tuple(json.loads((HERE/"suites.json").read_text())["suites"]))
    a.add_argument("--out",required=True)
    a.add_argument("--video",action="store_true")
    r=commands.add_parser("render")
    r.add_argument("--trace",type=Path,required=True)
    r.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="describe":
        print(json.dumps(describe(),ensure_ascii=True,indent=2,allow_nan=False))
    elif args.command=="evaluate":
        evaluate(args)
    else:
        render(json.loads(args.trace.read_text()),args.out)

if __name__=="__main__":
    main()
