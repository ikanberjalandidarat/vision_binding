"""Descriptive cross-run comparison with observed images; no inference."""
import argparse
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from mc_binding.vision_data import load_swaps
from mc_binding.io import digest,atomic_json


def compare(root,dataset):
    root,dataset=Path(root),Path(dataset)
    data,groups=load_swaps(dataset)
    results={};counts={};configs=[]
    for kind in ('residual','q','k','v'):
        run=root/f'vision-late-{kind}-vision-captures-04'
        if not (run/'status.json').exists():continue
        status=json.loads((run/'status.json').read_text())
        if status['state']!='complete':continue
        manifest=json.loads((run/'manifest.json').read_text());assert manifest['dataset_hash']==digest(data)
        config=manifest['config'].copy();assert config.pop('vision_patch_kind')==kind;configs.append(config)
        rows=[json.loads(s) for s in (run/'results.jsonl').read_text().splitlines()]
        saved=[r for p in (run/'families').glob('*.json') for r in json.loads(p.read_text())]
        assert len(rows)==status['rows']==552 and len({r['trial_key'] for r in rows})==552
        assert len(saved)==len(rows) and {r['trial_key']:r for r in saved}=={r['trial_key']:r for r in rows}
        results[kind]=[r for r in rows if r['condition']=='target']
        counts[kind]={'n':len(results[kind]),'specific_transfers':sum(r['target_transferred'] and r['neighbor_preserved'] for r in results[kind]),
          'median_target_log_odds_change':statistics.median(r['color_scores'][r['target_side']]['change_from_clean'] for r in results[kind])}
    assert results and all(c==configs[0] for c in configs)
    out=root/'vision-projection-comparison';out.mkdir(exist_ok=True)
    atomic_json(out/'summary.json',{'counts':counts,'scope':'Same calibration images; own-run baseline score changes. No independent-scene inference or necessity claim.'})
    fig=plt.figure(figsize=(13,8));grid=fig.add_gridspec(2,2,height_ratios=[1,1.5])
    for col,ctx in enumerate(('recipient','color_swap')):
        rec=next(r for r in groups['f0000'] if r['kind']=='pair' and r['context']==ctx)
        ax=fig.add_subplot(grid[0,col]);ax.imshow(Image.open(dataset/rec['image']));ax.axis('off');ax.set_title(ctx.replace('_',' ').title()+' — one of the four tested layouts')
    kinds=list(results);labels=['Whole state' if k=='residual' else k.upper()+' only' for k in kinds]
    ax=fig.add_subplot(grid[1,0]);vals=[counts[k]['specific_transfers'] for k in kinds];ax.bar(labels,vals,color=['#35866f' if k=='residual' else '#cc9762' for k in kinds]);ax.set_ylim(0,118);ax.set_ylabel('Target color transferred AND neighbor preserved');ax.set_title('Generated-answer changes')
    for i,k in enumerate(kinds):ax.text(i,vals[i]+2,f"{vals[i]}/{counts[k]['n']}",ha='center')
    ax=fig.add_subplot(grid[1,1]);vs=[[r['color_scores'][r['target_side']]['change_from_clean'] for r in results[k]] for k in kinds];ax.boxplot(vs,labels=labels,showfliers=True);ax.axhline(0,color='gray',linestyle='--');ax.set_title('How far preference moved toward the donor color');ax.set_ylabel('Change in donor-versus-original log odds')
    fig.suptitle('Does a Q, K or V edit reproduce the whole-state effect?',fontsize=19)
    fig.text(.04,.025,'Only completed local runs are shown. K/V absence is not a zero result. Patches differ in site and representation.\nScores use each run’s own baseline. Repeated layers/layouts are not independent scenes; boxes show descriptive distributions.',fontsize=10)
    fig.tight_layout(rect=[0,.075,1,.95]);fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    # Separate the depth sweep from the aggregate: a localized effect can be
    # obscured by pooling successful and ineffective intervention sites.
    sets=sorted({tuple(r['layer_set']) for r in next(iter(results.values()))},key=lambda s:(len(s)>1,s))
    per_layer=[];matrix=[]
    for layers in sets:
        entry={'layers':list(layers),'interventions':{}};values=[]
        for kind in kinds:
            subset=[r for r in results[kind] if tuple(r['layer_set'])==layers]
            n=len(subset);success=sum(r['target_transferred'] and r['neighbor_preserved'] for r in subset)
            assert n==8
            entry['interventions'][kind]={'specific_transfers':success,'n':n}
            values.append(success)
        per_layer.append(entry);matrix.append(values)
    atomic_json(out/'by-layer.json',{'results':per_layer,'definition':'Generated target answer matches donor color AND neighbor answer matches original color.'})
    fig=plt.figure(figsize=(12,11));grid=fig.add_gridspec(2,2,height_ratios=[1,3])
    for col,ctx in enumerate(('recipient','color_swap')):
        rec=next(r for r in groups['f0000'] if r['kind']=='pair' and r['context']==ctx)
        ax=fig.add_subplot(grid[0,col]);ax.imshow(Image.open(dataset/rec['image']));ax.axis('off')
        ax.set_title('Original image' if col==0 else 'Donor image: colors swapped')
    ax=fig.add_subplot(grid[1,:]);ax.imshow(matrix,cmap='YlGnBu',vmin=0,vmax=8,aspect='auto')
    ax.set_xticks(range(len(labels)),labels)
    ax.set_yticks(range(len(sets)),[' + '.join(map(str,s)) for s in sets])
    ax.set_ylabel('Vision layer(s) patched in one run — zero-based')
    ax.set_title('Successful color transfers / 8 cases at each site',pad=15)
    for row,values in enumerate(matrix):
        for col,value in enumerate(values):
            ax.text(col,row,f'{value}/8',ha='center',va='center',color='white' if value>=5 else '#203040',fontsize=13)
    fig.suptitle('V patching works mainly when layer 30 is included',fontsize=19)
    fig.text(.07,.025,'Success = target answer changes to donor color, while neighbor answer keeps its original color.\n'
             'The input pixels stay unchanged. Images above show one layout; each row tests 4 layouts × 2 target sides.\n'
             'Whole state: block output. Q/K/V: inside attention. These are different intervention sites.\n'
             'Same calibration scenes throughout; these counts are descriptive, not independent replication.',fontsize=10)
    fig.tight_layout(rect=[0,.11,1,.95]);fig.savefig(out/'layer-comparison.png',dpi=170);plt.close(fig)
    print(json.dumps(counts,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/oscar');p.add_argument('--dataset',default='runs/oscar/vision-captures-04');a=p.parse_args();compare(a.root,a.dataset)
