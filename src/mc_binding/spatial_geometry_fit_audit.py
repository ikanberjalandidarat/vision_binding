"""Training-fit/convergence audit; no test-based tuning or checkpoint selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .io import atomic_json
from .spatial_geometry_probes import fit_probe,project,pairs,load_cache

CHECKPOINTS=(0,10,50,100,200,500,1000)


def score(model,x,y):
    i,j,d=pairs(y);z=project(model,x)
    pred=np.full(len(i),model['constant']) if z is None else np.linalg.norm(z[i]-z[j],axis=1)
    result={'distance_mae':float(np.abs(pred-d).mean()),'distance_rmse':float(np.sqrt(np.mean((pred-d)**2)))}
    if model['method']=='ridge':result['coordinate_mae']=float(np.linalg.norm(z-y,axis=1).mean())
    return result


def audit_fit(x,y,xt,yt,method,k,steps=1000):
    curve=[]
    def observe(step,model):
        curve.append(dict(step=step,train=score(model,x,y),heldout=score(model,xt,yt)))
    model=fit_probe(x,y,method,k,seed=0,steps=steps,callback=observe,checkpoints=CHECKPOINTS)
    if not curve or curve[-1]['step']!=steps:
        observe(steps if method in ('orth','linear_metric') else 0,model)
    return curve


def positive_controls():
    rng=np.random.default_rng(13)
    x=rng.normal(size=(64,8))
    q=np.linalg.qr(rng.normal(size=(8,2)))[0]
    y=3*x@q
    return {'label':'NON-MINECRAFT synthetic linear spatial embedding; not a real model gate',
            'curves':{m:audit_fit(x[:40],y[:40],x[40:],y[40:],m,2) for m in ('ridge','orth','linear_metric')}}


def run(a):
    import torch
    torch.set_num_threads(2)
    a.output.mkdir(parents=True,exist_ok=False)
    try:
        atomic_json(a.output/'status.json',dict(state='running',stage='positive_controls'))
        controls=positive_controls();atomic_json(a.output/'positive-controls.json',controls)
        rows,features,manifest=load_cache(a.extraction,a.dataset)
        old=json.loads((a.previous/'report.json').read_text())
        if old['source_manifest']!=manifest or old['source_clips_sha256']!=hashlib.sha256((a.extraction/'clips.json').read_bytes()).hexdigest():
            raise ValueError('Previous result and current cache differ')
        lookup={(r['world'],r['train_route'],r['representation'],r['control'],r['method'],r['k']):r for r in old['results'] if r['seed']==0}
        y=np.asarray([r['position'] for r in rows],float)
        traces=[];reproduced=[]
        for world in sorted({r['geometry'] for r in rows}):
            for route in (0,1):
                tr=np.array(sorted([i for i,r in enumerate(rows) if r['geometry']==world and r['route']==route],key=lambda i:rows[i]['ticks'][-1]))
                te=np.array(sorted([i for i,r in enumerate(rows) if r['geometry']==world and r['route']==1-route],key=lambda i:rows[i]['ticks'][-1]))
                for name,x in features.items():
                    for control in ('true_positions','shifted_training_positions'):
                        train_y=y[tr] if control=='true_positions' else np.roll(y[tr],max(1,len(tr)//2),axis=0)
                        for method,k in [('mean_distance',0),('ridge',2),('orth',2),('orth',8),('linear_metric',2),('linear_metric',8)]:
                            curve=audit_fit(x[tr],train_y,x[te],y[te],method,k)
                            original=next(v for v in curve if v['step']==(200 if method in ('orth','linear_metric') else 0))
                            prior=lookup[(world,route,name,control,method,k)]['distance_mae_blocks']
                            delta=abs(original['heldout']['distance_mae']-prior)
                            reproduced.append(delta)
                            trace=dict(world=world,train_route=route,representation=name,control=control,method=method,k=k,seed=0,
                                       train_clips=len(tr),test_clips=len(te),curve=curve,prior_200_step_mae_difference=delta)
                            traces.append(trace)
                    atomic_json(a.output/'curves.json',traces)
                    atomic_json(a.output/'status.json',dict(state='running',completed_fits=len(traces)))
                    print(world,route,name,len(traces),flush=True)
        report=dict(state='complete',protocol='training_fit_audit_v1',fits=len(traces),seed=0,steps=list(CHECKPOINTS),
                    source_manifest=manifest,runtime={'torch':torch.__version__,'numpy':np.__version__},
                    prior_reproduction_max_mae_difference=max(reproduced),
                    prior_reproduction_all_close=bool(max(reproduced)<1e-5),
                    interpretation={'low_train_high_test':'Fit achieved but transfer fails under this sparse route protocol',
                                    'high_train':'Optimization, regularization or model-class limitations remain; not evidence of absent spatial information',
                                    'improvement_after_200':'Original training budget may have been insufficient; do not select best held-out checkpoint'},
                    limitations=['Test errors plotted for diagnosis only; no test-based tuning or checkpoint selection',
                                 'Only seed 0; other settings match original pilot',
                                 'Synthetic positive controls are NON-MINECRAFT; not evidence of real spatial information',
                                 'For shifted-label fits, training error is measured against shifted labels; held-out labels stay correct'],
                    results=traces)
        atomic_json(a.output/'report.json',report);atomic_json(a.output/'status.json',dict(state='complete',fits=len(traces)))
    except BaseException as exc:
        atomic_json(a.output/'status.json',dict(state='error',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('extraction','dataset','previous','output'):p.add_argument(name,type=Path)
    run(p.parse_args())
