"""Replay a saved model choice using a telemetry-guided controller.

This is a privileged-coordinate controller, not a learned navigation policy.
No model inference or ongoing visual patch is performed during movement.
"""
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from .io import atomic_json, digest, file_hash, environment, source_hash
from .vision_data import load_swaps
from .capture_pairs import clean_frame
from .backends.minestudio_smoke import json_value, verify_pose


def waypoint(obj):
    lo,hi=obj['bounds']
    return [(lo[0]+hi[0])/2, lo[2]-2.0]


def steering(pose, goal):
    x,z,yaw=(float(pose[k]) for k in ('x','z','yaw'))
    if not all(math.isfinite(v) for v in (x,z,yaw,*goal)):
        raise ValueError('Non-finite controller telemetry')
    dx,dz=goal[0]-x,goal[1]-z
    distance=math.hypot(dx,dz)
    desired=math.degrees(math.atan2(-dx,dz))
    turn=(desired-yaw+180)%360-180
    return distance, max(-10.,min(10.,turn)), int(abs(turn)<12 and distance>.8)


def approach(dataset, decisions, trial, output, max_steps=400):
    root,run,out=Path(dataset),Path(decisions),Path(output)
    data,groups=load_swaps(root)
    manifest=json.loads((run/'manifest.json').read_text())
    if manifest.get('experiment')!='destination_choice_v1' or manifest['dataset_hash']!=digest(data):
        raise ValueError('Decision/dataset provenance mismatch')
    if json.loads((run/'status.json').read_text())['state']!='complete':
        raise ValueError('Require completed destination experiment')
    rows=[json.loads(s) for s in (run/'results.jsonl').read_text().splitlines()]
    matches=[r for r in rows if r['trial_key']==trial]
    if len(matches)!=1 or matches[0]['choice_side'] not in (0,1):
        raise ValueError('Select exactly one valid LEFT/RIGHT trial')
    row=matches[0]
    record=next(r for r in groups[row['family']] if r['record_id']==row['record_id'])
    if record['kind']!='pair' or record['context']!=row['input_context']:
        raise ValueError('Decision input scene mismatch')
    if not 1<=max_steps<=2000:
        raise ValueError('max_steps must be 1–2000')
    out.mkdir(parents=True,exist_ok=False)
    report=dict(state='running',trial=row,dataset_hash=digest(data),decision_manifest_hash=digest(manifest),
                decision_results_sha256=file_hash(run/'results.jsonl'),environment=environment(),source_hash=source_hash(),
                controller='privileged waypoint from known world bounds; choice fixed before movement',
                scope='Open-loop destination decision, feedback movement via player telemetry; not closed-loop visual autonomy',trajectory=[])
    sim=None
    try:
        from minestudio.simulator import MinecraftSim
        from minestudio.simulator.callbacks import CommandsCallback
        sim=MinecraftSim(action_type='env',obs_size=(448,280),render_size=(448,280),seed=data['seed'])
        obs,info=sim.reset()
        obs,info=CommandsCallback(commands=['/gamemode creative','/forceload add -20 -5 20 25','/tp @p 0.5 200 0.5 0 0']).after_reset(sim,obs,info)
        def step(action):
            obs,_,done,truncated,info=sim.step(action)
            if done or truncated:
                raise RuntimeError('Simulator ended during approach')
            return obs,info
        for _ in range(200):obs,info=step(sim.noop_action())
        obs,info=CommandsCallback(commands=record['commands']).after_reset(sim,obs,info)
        for _ in range(200):obs,info=step(sim.noop_action())
        verify_pose(info,[.5,200,.5],0)
        Image.fromarray(np.asarray(obs['image'])).save(out/'start-raw.png')
        live,objects=clean_frame(obs['image'],record['objects']);live.save(out/'start.png')
        with Image.open(root/record['image']) as saved:
            error=float(np.abs(np.asarray(live,dtype=float)-np.asarray(saved,dtype=float)).mean())
        report['start_mean_pixel_error']=error
        # Fail closed if the rebuilt scene is materially different; threshold is
        # a render gate, not a claim that pixel equality proves world equality.
        if error>5 or any(max(abs(a-b) for a,b in zip(x['bbox'],y['bbox']))>2 for x,y in zip(objects,record['objects'])):
            raise RuntimeError('Rebuilt starting frame differs; inspect start images before movement')
        goals=[waypoint(o) for o in record['objects']]
        selected=row['choice_side'];goal=goals[selected]
        report['waypoints']=goals
        action=sim.noop_action()
        if 'forward' not in action or 'camera' not in action:
            raise RuntimeError('Unvalidated MineStudio movement action schema')
        reached=False
        for tick in range(max_steps+1):
            pose=json_value(info['player_pos'])
            distance,turn,forward=steering(pose,goal)
            if not 199.5<=float(pose['y'])<=200.5:
                raise RuntimeError('Unexpected controller elevation')
            entry=dict(tick=tick,pose=pose,distance_to_selected=distance)
            report['trajectory'].append(entry)
            if tick%20==0 or distance<=.8 or tick==max_steps:
                frame=f'frame-{tick:04d}.png';Image.fromarray(np.asarray(obs['image'])).save(out/frame);entry['frame']=frame
                atomic_json(out/'approach.json',report)
            if distance<=.8:
                reached=True;break
            if tick==max_steps:break
            action=sim.noop_action();action['camera']=np.array([0.,turn],dtype=np.float32);action['forward']=forward
            entry['action']=json_value(action)
            obs,info=step(action)
        expected=next(i for i,o in enumerate(record['objects']) if o['color']==row['goal_color'])
        report.update(state='complete',reached_selected_waypoint=reached,goal_side_in_rendered_scene=expected,
                      task_success=reached and selected==expected,steps=tick)
    except BaseException as exc:
        report.update(state='error',error=str(exc));raise
    finally:
        atomic_json(out/'approach.json',report)
        if sim is not None:sim.close()
