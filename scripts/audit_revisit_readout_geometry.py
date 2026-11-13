"""Recorded-coordinate oracle and pair audit; no encoder training or model selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from mc_binding.io import atomic_json, file_hash
from mc_binding.revisit_readouts import load, score


def run(features,readouts,out):
    rows,_,manifest,_=load(features)
    y=np.asarray([r['position'] for r in rows])
    report=json.loads((readouts/'report.json').read_text())
    assert report['source_manifest']==manifest
    train=np.array([i for i,r in enumerate(rows) if r['route'] in (0,1)])
    a,b=np.triu_indices(len(train),1);i,j=train[a],train[b]
    distances=np.linalg.norm(y[i]-y[j],axis=1)
    sites=np.array([r['site'] for r in rows]);headings=np.array([r['target_heading'] for r in rows])
    same=sites[i]==sites[j]
    angle=np.abs((headings[i]-headings[j]+180)%360-180)
    median=float(np.median(distances))
    oracle=[];baseline=[]
    for routes in ([2,3],[4,5]):
        for delta in (0,90,180):
            result=score(rows,y,routes,delta)
            oracle.append({k:result[k] for k in ('actual_routes','heading_difference','accuracy','mean_error_blocks','distance_mae_blocks','queries')})
            lookup={(r['route'],r['site'],r['target_heading']):k for k,r in enumerate(rows)}
            truth=[];same_truth=[];different_truth=[]
            for q in result['records']:
                qi=lookup[routes[q['query_route']-4],q['site'],q['query_heading']]
                for site in range(16):
                    gi=lookup[routes[q['gallery_route']-4],site,q['gallery_heading']]
                    d=float(np.linalg.norm(y[qi]-y[gi]));truth.append(d)
                    (same_truth if site==q['site'] else different_truth).append(d)
            baseline.append(dict(routes=routes,heading_difference=delta,constant_prediction_blocks=median,
                pair_distance_mae_blocks=float(np.abs(np.array(truth)-median).mean()),
                same_site_mae_blocks=float(np.abs(np.array(same_truth)-median).mean()),
                different_site_mae_blocks=float(np.abs(np.array(different_truth)-median).mean())))
    comparisons=[]
    for r in report['results']:
        base=next(v for v in baseline if v['routes']==[4,5] and v['heading_difference']==r['heading_difference'])
        comparisons.append(dict(feature=r['feature'],control=r['control'],family=r['family'],heading_difference=r['heading_difference'],
            readout_mae=r['distance_mae_blocks'],constant_mae=base['pair_distance_mae_blocks'],
            improvement_blocks=base['pair_distance_mae_blocks']-r['distance_mae_blocks']))
    result=dict(state='complete',source_report_sha256=file_hash(readouts/'report.json'),
        dataset_sha256=manifest['dataset_sha256'],
        coordinate_oracle_scope='Recorded pose used as features to test evaluator only; not a V-JEPA result',
        coordinate_oracle=oracle,
        training_pairs=dict(total=len(i),same_site=int(same.sum()),same_site_fraction=float(same.mean()),
            same_site_different_heading=int((same&(angle>0)).sum()),
            same_site_opposite_heading=int((same&(angle==180)).sum()),
            same_site_distance_range=[float(distances[same].min()),float(distances[same].max())],
            different_site=int((~same).sum()),weighting='All distinct training-row pairs equally weighted by metric losses'),
        constant_baseline=baseline,selected_readout_comparisons=comparisons,
        limits=['Geometry comparisons use retrieval-selected saved models, not geometry-selected models.',
               'Unselected candidate weights were not saved: separate geometry selection requires a new explicitly exploratory training run.',
               'A consistent permutation of site coordinates preserves identity across views; it is a geometry control, not a no-identity null.',
               'Coordinate oracle checks metadata and evaluator, not RGB/pose synchronization or feature extraction.',
               'Known sites and previously inspected test routes; no independent confirmation.'])
    out.mkdir(parents=True,exist_ok=True);atomic_json(out/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='selected_readout_comparisons'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('features',type=Path);p.add_argument('readouts',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.features,a.readouts,a.output)
