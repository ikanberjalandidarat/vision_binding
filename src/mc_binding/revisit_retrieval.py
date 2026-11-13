"""Predefined native-feature revisit tests; no learned transform or test tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from .io import atomic_json, digest, file_hash
from .revisit_video import protocol


def cosine(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    na=np.linalg.norm(a,axis=1,keepdims=True);nb=np.linalg.norm(b,axis=1,keepdims=True)
    if (na==0).any() or (nb==0).any():raise ValueError('Zero feature norm')
    return (a/na)@(b/nb).T


def evaluate(rows, features, delta, pixel_mse=False):
    """All queries both directions; exact nominal heading contrast, 16-site gallery."""
    lookup={(r['route'],r['site'],r['target_heading']):i for i,r in enumerate(rows)}
    records=[];pair_values={}
    offsets=[90,270] if delta==90 else [delta]
    for qr,gr in [(4,5),(5,4)]:
        for h in (0,90,180,270):
            for offset in offsets:
                gh=(h+offset)%360
                qi=[lookup[qr,s,h] for s in range(16)];gi=[lookup[gr,s,gh] for s in range(16)]
                if pixel_mse:
                    sims=-np.mean((features[qi,None,:]-features[None,gi,:])**2,axis=-1)
                else:
                    sims=cosine(features[qi],features[gi])
                winners=sims.argmax(1)
                physical=np.linalg.norm(np.array([rows[i]['position'] for i in qi])[:,None,:]-np.array([rows[i]['position'] for i in gi])[None,:,:],axis=-1)
                for s,g in enumerate(winners):
                    records.append(dict(query_route=qr,gallery_route=gr,site=s,predicted_site=int(g),
                        query_heading=h,gallery_heading=gh,correct=bool(s==g),
                        error_blocks=float(physical[s,g]),random_error_blocks=float(physical[s].mean()),
                        top_ties=int(np.isclose(sims[s],sims[s,g],rtol=0,atol=1e-12).sum())))
                    for t in range(16):
                        if s==t:continue
                        key=tuple(sorted((s,t)))
                        pair_values.setdefault(key,[]).append((1-sims[s,t],physical[s,t]))
    pairs=[dict(sites=list(k),cosine_distance=float(np.mean(v,axis=0)[0]),physical_distance=float(np.mean(v,axis=0)[1])) for k,v in sorted(pair_values.items())]
    rho=(float(spearmanr([p['cosine_distance'] for p in pairs],[p['physical_distance'] for p in pairs]).statistic) if np.ptp([p['cosine_distance'] for p in pairs])>0 else None)
    sites=[dict(site=s,accuracy=float(np.mean([r['correct'] for r in records if r['site']==s])),
        error_blocks=float(np.mean([r['error_blocks'] for r in records if r['site']==s]))) for s in range(16)]
    return dict(heading_difference=delta,queries=len(records),accuracy=float(np.mean([r['correct'] for r in records])),
        mean_error_blocks=float(np.mean([r['error_blocks'] for r in records])),
        random_error_blocks=float(np.mean([r['random_error_blocks'] for r in records])),
        directions={f'{qr}_to_{gr}':float(np.mean([r['correct'] for r in records if r['query_route']==qr])) for qr,gr in [(4,5),(5,4)]},
        per_site=sites,geometry=dict(distinct_unordered_site_pairs=len(pairs),spearman=rho,pairs=pairs),records=records)


def uncertainty(result):
    rng=np.random.default_rng(731);boot=rng.integers(0,16,(10000,16))
    acc=np.array([s['accuracy'] for s in result['per_site']]);err=np.array([s['error_blocks'] for s in result['per_site']])
    result['site_bootstrap_accuracy_95']=np.quantile(acc[boot].mean(1),[.025,.975]).tolist()
    result['site_bootstrap_error_95']=np.quantile(err[boot].mean(1),[.025,.975]).tolist()
    # Relabel route 5 globally: retain a site's identity across every heading and both directions.
    permutations=np.array([rng.permutation(16) for _ in range(10000)])
    null=np.zeros(10000)
    for r in result['records']:
        if r['gallery_route']==5:null+=(permutations[:,r['predicted_site']]==r['site'])
        else:null+=(r['predicted_site']==permutations[:,r['site']])
    null/=result['queries']
    result['permutation_p_one_sided']=(1+int((null>=result['accuracy']).sum()))/10001
    result['null_accuracy_mean']=float(null.mean())


def run(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'manifest.json').read_text());rows=json.loads((root/'clips.json').read_text())
    if json.loads((root/'status.json').read_text())!=dict(state='complete',clips=384):raise ValueError('Incomplete extraction')
    if manifest['protocol_sha256']!=digest(protocol()) or manifest['evaluation_protocol']!=protocol():raise ValueError('Protocol mismatch')
    if not manifest['frozen'] or manifest['pose_is_encoder_input']:raise ValueError('Not a frozen RGB-only run')
    expected={(r,s,h) for r in range(6) for s in range(16) for h in (0,90,180,270)}
    if len(rows)!=384 or {(r['route'],r['site'],r['target_heading']) for r in rows}!=expected:raise ValueError('Coverage mismatch')
    keys=[f'{c}_layer{l}' for c in ('video','repeat_last_frame') for l in (3,11,23)]
    features={k:[] for k in keys};hashes={}
    for r in rows:
        if r['split']!=('train','validation','test')[r['route']//2]:raise ValueError('Split mismatch')
        path=(root/r['cache']).resolve()
        if root.resolve() not in path.parents:raise ValueError('Unsafe cache path')
        hashes[r['cache']]=file_hash(path)
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!=set(keys):raise ValueError('Unexpected feature keys')
            for k in keys:
                if z[k].shape!=(1024,) or not np.isfinite(z[k]).all():raise ValueError('Invalid native feature vector')
                features[k].append(z[k])
    if len(hashes)!=384:raise ValueError('Repeated cache file')
    results=[]
    for k in keys:
        x=np.array(features[k])
        for delta in (180,0,90):
            result=evaluate(rows,x,delta);result['feature']=k
            if delta==180:uncertainty(result)
            results.append(result)
    primary=[r for r in results if r['heading_difference']==180]
    # Holm correction over all six predeclared primary condition/layer comparisons.
    ordered=sorted(primary,key=lambda r:r['permutation_p_one_sided']);last=0
    for rank,r in enumerate(ordered):
        last=max(last,min(1,(6-rank)*r['permutation_p_one_sided']));r['permutation_p_holm_six']=last
    deltas=[];boot=np.random.default_rng(731).integers(0,16,(10000,16))
    for l in (3,11,23):
        v=next(r for r in primary if r['feature']==f'video_layer{l}')
        b=next(r for r in primary if r['feature']==f'repeat_last_frame_layer{l}')
        d=np.array([s['accuracy'] for s in v['per_site']])-np.array([s['accuracy'] for s in b['per_site']])
        deltas.append(dict(layer_zero_based=l,video_minus_repeat_accuracy=float(d.mean()),site_bootstrap_95=np.quantile(d[boot].mean(1),[.025,.975]).tolist()))
    report=dict(state='complete_feature_tests_pixel_baseline_pending',subject=manifest['model'],
        protocol_sha256=manifest['protocol_sha256'],dataset_sha256=manifest['dataset_sha256'],
        checkpoint_sha256=manifest['checkpoint_sha256'],extraction_manifest_sha256=file_hash(root/'manifest.json'),
        feature_cache_hashes=hashes,chance_accuracy=1/16,results=results,temporal_differences=deltas,
        limitations=['Pixel baseline not run: exact raw endpoint images are not local.',
        'Bootstrap is conditional on 16 sites in one world and fixed routes; pairs/queries are correlated.',
        'Geometry aggregates each unordered site pair before rank correlation; no IID pair-level significance claimed.',
        'No supervised readout, fitted normalization, test layer selection, grid-cell or navigation claim.'])
    atomic_json(out/'report.json',report)
    small=[{k:v for k,v in r.items() if k not in ('records','geometry','per_site')} for r in results]
    atomic_json(out/'summary.json',dict(results=small,temporal_differences=deltas))
    plot(results,out)
    print(json.dumps(dict(results=small,temporal_differences=deltas),indent=2))


def pixel_run(root,dataset,out):
    from PIL import Image
    from .revisit_video import load_revisits, mask_ui
    root=Path(root);dataset=Path(dataset);out=Path(out)
    windows,provenance=load_revisits(dataset)
    manifest=json.loads((root/'manifest.json').read_text())
    if provenance['dataset_sha256']!=manifest['dataset_sha256'] or provenance['protocol_sha256']!=manifest['protocol_sha256']:
        raise ValueError('Dataset/protocol does not match extraction')
    x=[]
    for w in windows:
        f=w['frames'][-1];path=dataset/f['frame']
        if file_hash(path)!=f['sha256']:raise ValueError('Endpoint hash mismatch')
        with Image.open(path) as im:
            masked=Image.fromarray(mask_ui(np.asarray(im)))
            x.append(np.asarray(masked.resize((32,20),Image.Resampling.BILINEAR),dtype=float).ravel()/255)
    x=np.asarray(x);results=[]
    for delta in (180,0,90):
        r=evaluate(windows,x,delta,pixel_mse=True)
        # Geometric quantity is MSE, not cosine: relabel it explicitly.
        for pair in r['geometry']['pairs']:
            pair['pixel_mse']=pair.pop('cosine_distance')-1
        r['feature']='masked_endpoint_rgb_32x20'
        if delta==180:uncertainty(r)
        results.append(r)
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'report.json',dict(state='complete',dataset_sha256=provenance['dataset_sha256'],
        protocol_sha256=provenance['protocol_sha256'],results=results,
        implementation='Fixed-mask endpoint, bilinear resize to 32x20 RGB /255, nearest by MSE. No fitted parameters. Pixel resolution specified after inspecting feature results; no resolution sweep.',
        scope='Pixel appearance control; one world and fixed routes.'))
    for r in results:print(r['heading_difference'],r['accuracy'],r['mean_error_blocks'])


def plot(results,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(13,4.8),sharey=True)
    for ax,delta in zip(axs,[180,90,0]):
        for c,color,offset,label in [('video','#2072b5',-.18,'Video'),('repeat_last_frame','#dd8426',.18,'Repeated endpoint')]:
            rs=[next(r for r in results if r['feature']==f'{c}_layer{l}' and r['heading_difference']==delta) for l in (3,11,23)]
            vals=[r['accuracy']*100 for r in rs]
            ax.bar(np.arange(3)+offset,vals,.34,color=color,label=label)
            if delta==180:
                bounds=np.array([r['site_bootstrap_accuracy_95'] for r in rs]).T*100
                ax.errorbar(np.arange(3)+offset,vals,yerr=[np.array(vals)-bounds[0],bounds[1]-vals],fmt='none',color='black',capsize=3)
            for x,y in zip(np.arange(3)+offset,vals):ax.text(x,y+2,f'{y:.1f}',ha='center',fontsize=8)
        ax.axhline(6.25,color='gray',ls='--');ax.set_title(f'{delta}° heading difference'+ (' (primary)' if delta==180 else ' (secondary)'))
        ax.set_xticks(range(3));ax.set_xticklabels(['Block 4','Block 12','Block 24']);ax.set_ylim(0,110)
        ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Correct site retrieval (%)');axs[0].legend(loc='upper left',fontsize=8)
    fig.suptitle('Frozen V-JEPA: same-place recognition across held-out routes')
    fig.text(.05,.02,'16 candidates • Both route directions • Dashed: uniform chance 6.25% • Primary bars: site-cluster bootstrap 95%\nOne Minecraft world; pixel baseline pending. Same-heading success can reflect appearance and shared final approaches.',fontsize=9)
    fig.tight_layout(rect=[0,.1,1,.95]);fig.savefig(out/'retrieval.png',dpi=200);fig.savefig(out/'retrieval.svg');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('features');p.add_argument('output');p.add_argument('--pixel-dataset');a=p.parse_args()
    if a.pixel_dataset:pixel_run(a.features,a.pixel_dataset,a.output)
    else:run(a.features,a.output)
