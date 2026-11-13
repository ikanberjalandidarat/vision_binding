"""Exploratory supervised geometry readouts on the fixed six-route revisit pilot."""
import argparse
import json
from pathlib import Path
import numpy as np
from .io import atomic_json, digest, file_hash
from .revisit_video import protocol
from .revisit_retrieval import evaluate, uncertainty
from .spatial_geometry_probes import fit_probe, project

SPECS = [('ridge', 2, 1.), ('ridge', 2, 100.),
         ('orth', 2, None), ('orth', 8, None),
         ('linear_metric', 2, None), ('linear_metric', 8, None)]


def load(root):
    manifest = json.loads((root/'manifest.json').read_text())
    rows = json.loads((root/'clips.json').read_text())
    assert json.loads((root/'status.json').read_text()) == dict(state='complete', clips=384)
    assert manifest['protocol_sha256'] == digest(protocol())
    assert manifest['evaluation_protocol'] == protocol()
    assert manifest['frozen'] and not manifest['pose_is_encoder_input']
    assert len(rows) == 384
    assert {(r['route'], r['site'], r['target_heading']) for r in rows} == {
        (r,s,h) for r in range(6) for s in range(16) for h in (0,90,180,270)}
    keys = [f'{c}_layer{l}' for c in ('video','repeat_last_frame') for l in (3,11,23)]
    features = {k: [] for k in keys}; hashes = {}
    for r in rows:
        assert r['split'] == ('train','validation','test')[r['route']//2]
        p = (root/r['cache']).resolve()
        assert root.resolve() in p.parents and r['cache'] not in hashes
        hashes[r['cache']] = file_hash(p)
        with np.load(p, allow_pickle=False) as z:
            assert set(z.files) == set(keys)
            for k in keys:
                assert z[k].shape == (1024,) and np.isfinite(z[k]).all()
                features[k].append(z[k])
    return rows, {k: np.asarray(v, dtype=float) for k,v in features.items()}, manifest, hashes


def score(rows, x, routes, delta,constant=None):
    ids = [i for i,r in enumerate(rows) if r['route'] in routes]
    subset = [dict(rows[i], route=4+routes.index(rows[i]['route'])) for i in ids]
    result = evaluate(subset, x[ids], delta, pixel_mse=True)
    # evaluate's pixel option ranks Euclidean distances correctly but stores 1+MSE.
    # Recompute physical-distance errors BEFORE averaging across views/directions.
    lookup = {(r['route'],r['site'],r['target_heading']):i for i,r in enumerate(subset)}
    errors=[];baseline_errors=[]
    for q in result['records']:
        i=lookup[q['query_route'],q['site'],q['query_heading']]
        for s in range(16):
            j=lookup[q['gallery_route'],s,q['gallery_heading']]
            pred=float(np.linalg.norm(x[ids[i]]-x[ids[j]]))
            true=float(np.linalg.norm(np.array(subset[i]['position'])-subset[j]['position']))
            errors.append(abs(pred-true))
            if constant is not None:baseline_errors.append(abs(constant-true))
    result['distance_mae_blocks']=float(np.mean(errors))
    if baseline_errors:result['constant_distance_mae_blocks']=float(np.mean(baseline_errors))
    result.pop('geometry')  # Do not mislabel MSE as cosine or Euclidean distance.
    result['actual_routes']=routes
    result['route_key_note']='4/5 in records denote first/second actual_routes entry'
    return result


def balanced_pair_weights(rows):
    """Half the loss on same-site/different-heading; half on different sites."""
    i,j=np.triu_indices(len(rows),1)
    site=np.array([r['site'] for r in rows]);h=np.array([r['target_heading'] for r in rows])
    positive=(site[i]==site[j])&(h[i]!=h[j])
    negative=site[i]!=site[j]
    if not positive.any() or not negative.any():raise ValueError('Missing pair strata')
    w=np.zeros(len(i));w[positive]=.5/positive.sum();w[negative]=.5/negative.sum()
    return w


def fit(x,y,spec,steps,pair_weights=None):
    method,k,alpha=spec
    model=fit_probe(x,y,method,k,seed=731,steps=steps,pair_weights=pair_weights)
    if method=='ridge':
        a=(x-model['center'])/model['scale']
        model['w']=a.T@np.linalg.solve(a@a.T+alpha*np.eye(len(a)),y-y.mean(0))
    return model


def run(root,out,steps=300,balanced=False,selection="retrieval"):
    import torch
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'status.json',dict(state='running'))
    try:
        rows,features,manifest,hashes=load(root)
        y=np.array([r['position'] for r in rows],float)
        train=np.array([i for i,r in enumerate(rows) if r['route'] in (0,1)])
        assert y.shape==(384,2) and np.isfinite(y).all()
        # One site permutation shared across training routes and headings.
        perm=np.random.default_rng(731).permutation(16)
        lookup={(r['route'],r['site'],r['target_heading']):i for i,r in enumerate(rows)}
        shuffled=np.array([y[lookup[rows[i]['route'],int(perm[rows[i]['site']]),rows[i]['target_heading']]] for i in train])
        weights=balanced_pair_weights([rows[i] for i in train]) if balanced else None
        ti,tj=np.triu_indices(len(train),1)
        constant=float(np.median(np.linalg.norm(y[train[ti]]-y[train[tj]],axis=1)))
        selections=[]; results=[]; diagnostics=[]
        for key,x in features.items():
            for control,labels in [('true_positions',y[train]),('permuted_training_sites',shuffled)]:
                for family in ('ridge','orth','linear_metric'):
                    candidates=[]
                    for spec in [s for s in SPECS if s[0]==family]:
                        model=fit(x[train],labels,spec,steps,weights)
                        z=project(model,x)
                        val=score(rows,z,[2,3],180)
                        candidates.append((val['accuracy'],-val['distance_mae_blocks'],spec,model))
                        d=np.linalg.norm(z[train,None]-z[None,train],axis=-1)
                        t=np.linalg.norm(labels[:,None]-labels[None,:],axis=-1)
                        mask=~np.eye(len(train),dtype=bool)
                        diagnostics.append(dict(feature=key,control=control,spec=spec,
                            training_pair_mae=float(np.abs(d-t)[mask].mean()),
                            validation_opposite_accuracy=val['accuracy'],validation_distance_mae=val['distance_mae_blocks']))
                    # Validation only; deterministic ordering breaks remaining ties.
                    best=max(candidates,key=lambda c:(c[0],c[1]) if selection=='retrieval' else (c[1],c[0]))
                    spec,model=best[2:]
                    tag=f'{key}-{control}-{family}'
                    np.savez_compressed(out/(tag+'.npz'),**model)
                    selections.append(dict(feature=key,control=control,family=family,spec=spec,
                        validation_accuracy=best[0],validation_distance_mae=-best[1],artifact=tag+'.npz'))
                    z=project(model,x)
                    for delta in (180,90,0):
                        r=score(rows,z,[4,5],delta,constant=constant)
                        uncertainty(r)
                        # Exploratory, multiple comparisons: no uncorrected significance claims.
                        r.pop('permutation_p_one_sided');r.pop('null_accuracy_mean')
                        results.append(dict(r,feature=key,control=control,family=family,spec=spec))
                    atomic_json(out/'progress.json',dict(last=tag,selected_readouts=len(selections)))
                    print(tag, spec, flush=True)
        contrasts=[]
        boot=np.random.default_rng(731).integers(0,16,(10000,16))
        for layer in (3,11,23):
            for family in ('ridge','orth','linear_metric'):
                for delta in (180,90,0):
                    matched=[next(r for r in results if r['feature']==f'{c}_layer{layer}' and r['control']=='true_positions' and r['family']==family and r['heading_difference']==delta) for c in ('video','repeat_last_frame')]
                    d=np.array([s['accuracy'] for s in matched[0]['per_site']])-np.array([s['accuracy'] for s in matched[1]['per_site']])
                    contrasts.append(dict(layer=layer,family=family,heading_difference=delta,
                        video_minus_repeat_accuracy=float(d.mean()),site_bootstrap_95=np.quantile(d[boot].mean(1),[.025,.975]).tolist()))
        report=dict(state='complete',scope='Exploratory known-site readout evaluation; test routes previously inspected',
            training_routes=[0,1],validation_routes=[2,3],test_routes=[4,5],
            selection=selection,
            selection_rule='Validation 180-degree only; retrieval maximizes accuracy then minimizes distance MAE; geometry reverses priorities',
            balanced_pairs=balanced,
            pair_loss='50% same-site/different-heading + 50% different-site; same-site/same-heading excluded' if balanced else 'Uniform distinct pairs',
            ridge_note='Coordinate ridge is unchanged by pair weights; it has a pointwise objective',
            constant_distance_prediction_blocks=constant,
            steps=steps,seed=731,permuted_sites=perm.tolist(),source_manifest=manifest,cache_hashes=hashes,
            selections=selections,training_diagnostics=diagnostics,results=results,temporal_contrasts=contrasts,
            limits=['Pose supervises readouts only; encoder stays frozen.',
                '16 known sites, one world, fixed routes. Not unseen-location or independent confirmation.',
                'Site bootstrap intervals are conditional and unadjusted for multiple comparisons.',
                'Distance MAE includes same-site pairs and is averaged over correlated query/gallery comparisons.',
                'No evidence of temporal advantage without video outperforming repeated frames.',
                'Fresh routes and held-out locations required for confirmation. No grid-cell or navigation claim.'])
        atomic_json(out/'report.json',report)
        atomic_json(out/'summary.json',dict(scope=report['scope'],balanced_pairs=balanced,selection=selection,selections=selections,temporal_contrasts=contrasts,
            results=[{k:v for k,v in r.items() if k not in ('records','per_site')} for r in results]))
        atomic_json(out/'status.json',dict(state='complete',selected_readouts=len(selections)))
    except Exception as exc:
        atomic_json(out/'status.json',dict(state='error',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('features',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--balanced',action='store_true')
    p.add_argument('--selection',choices=['retrieval','geometry'],default='retrieval')
    a=p.parse_args();run(a.features,a.output,a.steps,a.balanced,a.selection)
