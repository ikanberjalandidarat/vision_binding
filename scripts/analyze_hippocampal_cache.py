"""Offline exploratory audit of cached Minecraft memory; no training or rendering."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
import torch
from mc_binding.hippocampal import SpatialMemory, odometry, rate_maps


def correlations(a, b):
    """Per-unit Pearson r; undefined constant units stay null."""
    if len(a) < 3:
        return [None] * a.shape[1]
    a = a - a.mean(0); b = b - b.mean(0)
    den = np.sqrt((a*a).sum(0) * (b*b).sum(0))
    return [float(v/d) if d > 1e-10 else None for v,d in zip((a*b).sum(0),den)]


def matched_pairs(poses_a, poses_b=None, heading='same'):
    """One nearest eligible reference per query; references may repeat, no IID claim."""
    within = poses_b is None
    poses_b = poses_a if within else poses_b
    xy = lambda ps: np.array([[p['x'],p['z']] for p in ps])
    dist = np.linalg.norm(xy(poses_a)[:,None]-xy(poses_b)[None,:],axis=-1)
    yaw_a = np.array([p['yaw'] for p in poses_a]); yaw_b = np.array([p['yaw'] for p in poses_b])
    dy = abs((yaw_a[:,None]-yaw_b[None,:]+180)%360-180)
    valid = (dist < .6) & ((dy < 25) if heading=='same' else (dy > 90))
    if within:
        valid &= np.arange(len(poses_b))[None,:] <= np.arange(len(poses_a))[:,None]-12
    pairs=[]
    for i in range(len(poses_a)):
        js=np.flatnonzero(valid[i])
        if len(js):pairs.append((i,int(js[np.argmin(dist[i,js])])))
    return np.array(pairs,dtype=int).reshape(-1,2)


def intervention(model, x, g, mode):
    """Keep learned motion states; replace retrieval only, using strictly past values."""
    outputs=[]
    for t in range(len(x)):
        recalled = x[:t].mean(0) if t and mode=='past_mean' else model.empty
        p=torch.tanh(model.conjunction(torch.cat([g[t],recalled])))
        outputs.append(model.readout(p))
    return torch.stack(outputs)


def main(root):
    torch.set_num_threads(4)
    entries=json.loads((root/'sequences.json').read_text())
    data=torch.load(root/'features.pt',map_location='cpu',weights_only=True)
    checkpoint=torch.load(root/'memory.pt',map_location='cpu',weights_only=True)
    lookup={e['episode']:e for e in data['sequences']}
    nets={}
    for label,key in [('learned','state'),('untrained','initial')]:
        net=SpatialMemory();net.load_state_dict(checkpoint[key]);net.eval();nets[label]=net
    acts={}; errors=[]; pairscores=[]; maprows=[]; saturation=[]
    with torch.no_grad():
        for e in entries:
            name=e['episode']; c=lookup[name]; poses=[f['pose'] for f in e['frames']]
            assert len(c['x'])==len(poses) and torch.allclose(c['motion'],odometry(poses),atol=1e-6)
            x=(c['x']-checkpoint['mean'])/checkpoint['scale']
            assert torch.isfinite(x).all()
            split=e['split'] if e['split']!='train' or e['route']==0 else 'heldout_route'
            for label,net in nets.items():
                pred,g,p,_=net(x,c['motion']);acts[name,label]=(g.numpy(),p.numpy())
                candidates={label:pred}
                if label=='learned':
                    candidates.update({mode:intervention(net,x,g,mode) for mode in ('empty_retrieval','past_mean')})
                    saturation.append(dict(episode=name,p_fraction_abs_gt_099=float((p.abs()>.99).float().mean())))
                for mode,y in candidates.items():
                    errors.append(dict(episode=name,split=split,condition=mode,mse=float((y[1:]-x[1:]).square().mean())))
                for heading in ('same','different'):
                    pairs=matched_pairs(poses,heading=heading)
                    for k,pop in enumerate(('g','p')):
                        a=acts[name,label][k]
                        pairscores.append(dict(episode=name,model=label,population=pop,comparison='within_route',heading=heading,n=len(pairs),r=correlations(a[pairs[:,0]],a[pairs[:,1]])))
        for e in entries:
            if e['route']!=0:continue
            other=next(v for v in entries if v['family']==e['family'] and v['route']==1)
            pa=[f['pose'] for f in e['frames']];pb=[f['pose'] for f in other['frames']]
            for label in nets:
                for k,pop in enumerate(('g','p')):
                    a=acts[e['episode'],label][k];b=acts[other['episode'],label][k]
                    ma,ca=rate_maps(a,pa,[-18,18,-5,27]);mb,cb=rate_maps(b,pb,[-18,18,-5,27]);mask=(ca>=3)&(cb>=3)
                    maprows.append(dict(family=e['family'],split=e['split'],model=label,population=pop,common_bins=int(mask.sum()),r=correlations(ma[mask],mb[mask])))
                    for heading in ('same','different'):
                        pairs=matched_pairs(pa,pb,heading)
                        pairscores.append(dict(episode=e['episode'],model=label,population=pop,comparison='across_routes',heading=heading,n=len(pairs),r=correlations(a[pairs[:,0]],b[pairs[:,1]])))
    out=root/'analysis';out.mkdir(exist_ok=True)
    result=dict(scope='Exploratory single-seed cached audit, scripted Minecraft trajectories and privileged odometry. Correlated frames and shared routes; not independent trials or biological cell evidence.',thresholds=dict(radius_blocks=.6,min_frame_gap=12,same_heading_degrees=25,different_heading_degrees=90,map_min_occupancy=3),prediction=errors,matched_visits=pairscores,cross_route_maps=maprows,saturation=saturation)
    (out/'cache-audit.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    groups=['train','heldout_route','validation','test'];modes=['learned','past_mean','empty_retrieval','untrained']
    for j,mode in enumerate(modes):
        vals=[np.mean([v['mse'] for v in errors if v['condition']==mode and v['split']==s]) for s in groups]
        axes[0].bar(np.arange(4)+(j-1.5)*.19,vals,.19,label=mode.replace('_',' '))
    axes[0].set(xticks=np.arange(4),xticklabels=groups,ylabel='Normalized feature prediction MSE',title='Does learned retrieval help?');axes[0].legend(fontsize=8)
    labels=[];values=[]
    for pop in ('g','p'):
        for model in nets:
            labels.append(pop+' / '+model)
            values.append([v for row in maprows if row['population']==pop and row['model']==model for v in row['r'] if v is not None])
    axes[1].boxplot(values,labels=labels,showfliers=False);axes[1].axhline(0,color='gray',lw=.8)
    axes[1].set(ylabel='Cross-route spatial-map Pearson r',ylim=(-1,1),title='Same world bins, different routes')
    fig.suptitle('Minecraft memory audit • all 32 units • one training seed',fontsize=13)
    fig.savefig(out/'cache-audit.png',dpi=180);fig.savefig(out/'cache-audit.svg');plt.close(fig)
    table=''.join('<tr><td>'+s+'</td>'+''.join(f'<td>{np.mean([v["mse"] for v in errors if v["condition"]==m and v["split"]==s]):.3f}</td>' for m in modes)+'</tr>' for s in groups)
    visit_table=''
    for comp in ('within_route','across_routes'):
        for heading in ('same','different'):
            medians=[]
            for pop in ('g','p'):
                rows=[r for r in pairscores if r['model']=='learned' and r['comparison']==comp and r['heading']==heading and r['population']==pop]
                values=[v for r in rows for v in r['r'] if v is not None]
                medians.append(f'{np.median(values):.3f}' if values else 'undefined')
            counts=[r['n'] for r in rows]
            visit_table+=f'<tr><td>{comp}</td><td>{heading}</td><td>{min(counts)}–{max(counts)}</td><td>{medians[0]}</td><td>{medians[1]}</td></tr>'
    findings='<h2>What the reconstructed activations show</h2><p>Learned retrieval improves held-out sensory prediction relative to both replacement controls. However, spatial-map correlations across opposite routes are negative at the population median. Repetition of the same route and heading gives much stronger consistency. This supports useful route/view-conditioned memory, but does not establish heading-invariant place codes or grid cells.</p><table><tr><th>Matching</th><th>Heading</th><th>Pairs per sequence/family</th><th>Median g r</th><th>Median p r</th></tr>'+visit_table+'</table><p>Medians pool unit correlations across sequences; these are descriptive, not independent samples. Nearest reference frames can be reused. Across-route same-heading matches are particularly sparse. Position, heading and route phase remain confounded, so these results cannot attribute the effect to heading alone.</p>'
    (out/'cache-audit.html').write_text('''<!doctype html><meta charset="utf-8"><title>Memory cache audit</title><style>body{font:17px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:20px}img{width:100%}td,th{padding:10px;text-align:left}code{overflow-wrap:anywhere}</style><h1>Does the memory encode stable world locations?</h1><p>Offline reconstruction of learned and untrained units, using the saved model and frozen Qwen feature cache. No new training or navigation.</p><img src="cache-audit.png"><p><code>'''+html.escape(str(out/'cache-audit.png'))+'''</code></p><h2>Prediction controls</h2><table><tr><th>Split</th><th>Learned</th><th>Past mean</th><th>Empty retrieval</th><th>Untrained</th></tr>'''+table+'''</table><p>Each entry averages episode MSE. Past mean replaces attention with uniform averaging of strictly past observations. Empty retrieval supplies the learned empty vector at every step. These are inference interventions, not separately trained baselines.</p>'''+findings+'''<h2>Interpretation limits</h2><p>Cross-route maps average over heading; matched-visit results in the JSON separate heading differences. Correlations alone do not distinguish position, route phase, and heading. Untrained units are essential controls. No significance tests treat frames or units as independent replications.</p><p>The g state receives motion only, not visual observations. It cannot visually correct localization drift in this implementation. The p state combines g with retrieved past visual features. Neither name establishes place or grid cells. Six geometries share scripted paths; only one model seed was trained.</p><p><a href="cache-audit.json">All units, matched visits, saturation and metrics (JSON)</a> · <a href="cache-audit.svg">Vector figure</a></p>''')
    print(out/'cache-audit.html')
    print(json.dumps({s:{m:round(float(np.mean([v['mse'] for v in errors if v['condition']==m and v['split']==s])),4) for m in modes} for s in groups},indent=2))
    for pop in ('g','p'):
        for model in nets:
            vals=[v for row in maprows if row['population']==pop and row['model']==model for v in row['r'] if v is not None]
            print(pop,model,'median map r',np.median(vals))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);main(p.parse_args().run)
