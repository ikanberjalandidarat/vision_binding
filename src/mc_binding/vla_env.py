"""MineStudio JSON-lines worker. Telemetry/teacher remain outside policy inputs."""
import contextlib
import json
import math
import sys
import traceback
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .io import atomic_json
from .approach import steering, waypoint
from .backends.minestudio_smoke import json_value, verify_pose

ACTIONS=('forward','turn_left','turn_right','stop')


def teacher_action(pose,goal):
    if goal is None: return 3
    distance,turn,forward=steering(pose,goal)
    if distance<=.8: return 3
    return 0 if forward else 1 if turn<0 else 2


def action_dict(sim,index):
    if type(index) is not int or not 0<=index<len(ACTIONS): raise ValueError('Invalid action')
    a=sim.noop_action()
    if 'camera' not in a or 'forward' not in a: raise RuntimeError('Unsupported movement action schema')
    a['forward']=int(index==0)
    a['camera']=np.array([0.,-5. if index==1 else 5. if index==2 else 0.],np.float32)
    return a


class Environment:
    def __init__(self,output,chunk=2):
        if not 1<=chunk<=5: raise ValueError('Action chunk must be 1..5 ticks')
        self.root=Path(output).resolve();self.root.mkdir(parents=True,exist_ok=False)
        self.chunk=chunk;self.sim=None;self.episode=None
    def reset(self,record,goal,seed,episode,reference,training_start=None,diagnostic_start=None):
        if diagnostic_start is not None:
            if training_start is not None or not episode.startswith('probe-'):
                raise ValueError('Near-start evaluation must be explicitly labeled probe')
        from minestudio.simulator import MinecraftSim
        from minestudio.simulator.callbacks import CommandsCallback
        if self.sim is None:
            self.sim=MinecraftSim(action_type='env',obs_size=(448,280),render_size=(448,280),seed=seed)
            self.obs,self.info=self.sim.reset()
        # Bootstrap at ground height first, avoiding collision with previous structures.
        cmds=['/gamemode creative','/forceload add -20 -5 20 25','/tp @p 0.5 200 0.5 0 0']
        self.obs,self.info=CommandsCallback(commands=cmds).after_reset(self.sim,self.obs,self.info)
        for _ in range(200): self._step(self.sim.noop_action())
        self.obs,self.info=CommandsCallback(commands=record['commands']).after_reset(self.sim,self.obs,self.info)
        for _ in range(200): self._step(self.sim.noop_action())
        verify_pose(self.info,[.5,200,.5],0)
        from .capture_pairs import CROP,CROSSHAIR,CLEANUP
        frame=Image.fromarray(np.asarray(self.obs['image']))
        a,b,c,d=CROSSHAIR
        ImageDraw.Draw(frame).rectangle((a,b,c-1,d-1),fill=tuple(CLEANUP['crosshair_fill_rgb']))
        frame=frame.crop(CROP)
        with Image.open(reference) as saved:
            if saved.size!=frame.size:raise ValueError('Start reference size mismatch')
            error=float(np.abs(np.asarray(frame,dtype=float)-np.asarray(saved,dtype=float)).mean())
        if error>5:
            frame.save(self.root/(episode+'-failed-reset.png'))
            raise RuntimeError(f'Rebuilt scene differs from reviewed capture: MAE={error}')
        matches=[o for o in record['objects'] if (o['color'],o['type'])==tuple(goal)]
        if len(matches)>1: raise ValueError('Ambiguous goal')
        self.goal=waypoint(matches[0]) if matches else None
        start=diagnostic_start if diagnostic_start is not None else training_start
        if start is not None:
            if diagnostic_start is None and not episode.startswith('train-'):
                raise ValueError('Curriculum starts are forbidden during evaluation')
            distance=float(start['distance'])
            if not 1. <= distance <= 6.:raise ValueError('Unsafe curriculum distance')
            anchor=self.goal if self.goal is not None else waypoint(record['objects'][start['anchor_index']])
            position=[anchor[0],200.,anchor[1]-distance]
            # Retain the original scene reconstruction gate, then move only the player.
            self.obs,self.info=CommandsCallback(commands=[f'/tp @p {position[0]} 200 {position[2]} 0 0']).after_reset(self.sim,self.obs,self.info)
            for _ in range(20):self._step(self.sim.noop_action())
            verify_pose(self.info,position,0)
        self.episode=self.root/episode;self.episode.mkdir(exist_ok=False)
        self.trace=[];self.frames=[];self.tick=0
        self.metadata=dict(record_id=record['record_id'],goal=goal,goal_present=self.goal is not None,objects=record['objects'],waypoint=self.goal,start_pixel_error=error,training_start=training_start,diagnostic_start=diagnostic_start)
        return self.observe()
    def _step(self,a):
        self.obs,_,done,truncated,self.info=self.sim.step(a)
        if isinstance(self.info,dict) and 'error' in self.info:raise RuntimeError('Simulator returned error/fallback observation: '+str(self.info['error']))
        if done or truncated: raise RuntimeError('Simulator terminated')
    def observe(self):
        pose=json_value(self.info['player_pos'])
        if not 199.5<=float(pose['y'])<=200.5: raise RuntimeError('Unexpected elevation')
        name=f'frame-{len(self.trace):04d}.png';p=self.episode/name
        image=Image.fromarray(np.asarray(self.obs['image']));image.save(p)
        self.trace.append(dict(tick=self.tick,pose=pose,frame=name))
        annotated=Image.new('RGB',(448,310),'white');annotated.paste(image,(0,30))
        ImageDraw.Draw(annotated).text((5,5),f"t={self.tick} X={pose['x']:.2f} Y={pose['y']:.2f} Z={pose['z']:.2f}",fill='black')
        self.frames.append(annotated)
        # This is the entire observation interface. No coordinates, boxes or target IDs.
        return {'frame':str(p)}
    def teacher(self): return {'action':teacher_action(json_value(self.info['player_pos']),self.goal)}
    def step(self,index):
        if index==3: raise ValueError('Use finish for stop')
        self.trace[-1]['action']=ACTIONS[index]
        for _ in range(self.chunk): self._step(action_dict(self.sim,index));self.tick+=1
        return self.observe()
    def finish(self,stopped):
        pose=json_value(self.info['player_pos'])
        distance=None if self.goal is None else math.hypot(float(pose['x'])-self.goal[0],float(pose['z'])-self.goal[1])
        from .navigation_metrics import score_episode
        metrics=score_episode(self.trace,self.goal,stopped);success=metrics['success']
        result=dict(state='complete',**self.metadata,stopped=stopped,**metrics,trajectory=self.trace,
                    scope='Privileged state used only by teacher/evaluator; learned policy receives RGB and instruction')
        atomic_json(self.episode/'episode.json',result)
        self.frames[0].save(self.episode/'movement.gif',save_all=True,append_images=self.frames[1:],duration=self.chunk*50,loop=0)
        canvas=Image.new('RGB',(600,600),'white');d=ImageDraw.Draw(canvas)
        points=[(300-float(t['pose']['x'])*12,550-float(t['pose']['z'])*20) for t in self.trace]
        if len(points)>1:d.line(points,fill='purple',width=3)
        for o in self.metadata['objects']:
            lo,hi=o['bounds'];x=300-(lo[0]+hi[0])*6;y=550-lo[2]*20
            d.ellipse((x-6,y-6,x+6,y+6),fill=o['color']);d.text((x+8,y),o['type'],fill='black')
        d.text((10,10),'Trajectory: -X right, +Z up (initial camera axes)',fill='black');canvas.save(self.episode/'trajectory.png')
        (self.episode/'report.html').write_text('<!doctype html><meta charset="utf-8"><h1>VLA episode</h1><p>Success: '+str(success)+'</p><img src="movement.gif"><img src="trajectory.png"><p>GIF includes raw HUD. Coordinates are evaluator annotations, not policy input.</p>')
        return dict(metrics,episode_file=str(self.episode/'episode.json'),goal_present=self.goal is not None)
    def reward_distance(self):
        pose=json_value(self.info['player_pos'])
        return None if self.goal is None else math.hypot(float(pose['x'])-self.goal[0],float(pose['z'])-self.goal[1])
    def rl_step(self,index,last=False,mode='sparse',gamma=.99):
        from .rl_rewards import transition_reward
        before=self.reward_distance()
        observation={} if index==3 else self.step(index)
        terminal=bool(index==3 or last)
        after=self.reward_distance()
        result=self.finish(index==3) if terminal else None
        reward=transition_reward(before,after,terminal,bool(result and result['success']),mode,gamma)
        # Only scalar reward, terminal flag and RGB path cross into RL collection.
        # Coordinates and object identities remain in simulator/evaluation files.
        return dict(observation=observation,reward=reward,terminal=terminal,
                    success=bool(result['success']) if terminal else None)

    def close(self):
        if self.sim is not None:
            self.sim.close();self.sim=None


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--chunk',type=int,default=2);args=p.parse_args()
    env=Environment(args.output,args.chunk)
    try:
        for line in sys.stdin:
            try:
                request=json.loads(line);cmd=request.pop('command')
                with contextlib.redirect_stdout(sys.stderr):
                    result=getattr(env,cmd)(**request) if cmd in ('reset','teacher','step','finish','close','rl_step') else None
                if result is None and cmd!='close': raise ValueError('Unknown worker command')
                print(json.dumps({'ok':True,'result':result}),flush=True)
                if cmd=='close':break
            except BaseException as e:
                traceback.print_exc(file=sys.stderr)
                print(json.dumps({'ok':False,'error':repr(e)}),flush=True)
                break
    finally:
        with contextlib.redirect_stdout(sys.stderr):env.close()

if __name__=='__main__':main()
