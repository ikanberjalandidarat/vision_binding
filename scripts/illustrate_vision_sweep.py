"""ELI5 figures using observed Minecraft frames and saved behavioral results."""
import argparse
import base64
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from mc_binding.vision_data import load_swaps
from mc_binding.io import digest


def illustrate(run,dataset):
    run,dataset=Path(run),Path(dataset)
    data,groups=load_swaps(dataset)
    assert digest(data)==json.loads((run/'manifest.json').read_text())['dataset_hash']
    rows=[json.loads(s) for s in (run/'results.jsonl').read_text().splitlines()]
    trials=[r for r in rows if r['condition']=='target']
    out=run/'analysis';out.mkdir(exist_ok=True)
    pair={r['context']:r for r in groups['f0000'] if r['kind']=='pair'}
    fig,axes=plt.subplots(2,3,figsize=(16,8),gridspec_kw={'height_ratios':[1,1.15]})
    images=[dataset/pair['recipient']['image'],dataset/pair['color_swap']['image'],run/'alignment/f0000/patch-target-target0.png']
    titles=['1. Original picture: blue arch + red tower','2. Donor picture: red arch + blue tower','3. Copy internal numbers at orange cells']
    for ax,path,title in zip(axes[0],images,titles):
        ax.imshow(Image.open(path));ax.axis('off');ax.set_title(title,fontsize=12)
    examples=[(15,0),(31,0),(31,1)]
    for ax,(layer,target) in zip(axes[1],examples):
        r=next(r for r in trials if r['family']=='f0000' and r['layer_set']==[layer] and r['target_side']==target)
        ax.axis('off')
        text=f"Layer {layer} | edit {'arch (left)' if target==0 else 'tower (right)'}\n\nLeft answer: {r['answers'][0]}\nRight answer: {r['answers'][1]}\n\nTarget changed to donor color: {'YES' if r['target_transferred'] else 'NO'}\nNeighbor kept original color: {'YES' if r['neighbor_preserved'] else 'NO'}"
        ax.text(.05,.92,text,va='top',fontsize=16,linespacing=1.65,bbox=dict(boxstyle='round,pad=.8',facecolor='#edf5f1',edgecolor='#b0ccc1'))
    fig.suptitle('We change the model’s internal numbers—not the Minecraft pixels',fontsize=20)
    fig.text(.03,.015,'Actual saved images and answers. Orange = edited token positions, not an attention map. No training or pixel recoloring.',fontsize=12)
    fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(out/'eli5-how-it-works.png',dpi=160);plt.close(fig)
    fig=plt.figure(figsize=(16,11));grid=fig.add_gridspec(3,4,height_ratios=[1,1.35,1.35])
    for i,(fid,records) in enumerate(sorted(groups.items())):
        ax=fig.add_subplot(grid[0,i]);rec=next(r for r in records if r['context']=='recipient' and r['kind']=='pair')
        ax.imshow(Image.open(dataset/rec['image']));ax.axis('off');ax.set_title(fid+' original picture',fontsize=12)
    ax=fig.add_subplot(grid[1,:]);single=[]
    for l in range(32):single.append(sum(r['target_transferred'] and r['neighbor_preserved'] for r in trials if r['layer_set']==[l]))
    ax.bar(range(32),single,color=['#31866b' if v==8 else '#d58d62' for v in single]);ax.set_xticks(range(32));ax.set_ylim(0,9);ax.set_yticks([0,2,4,6,8]);ax.set_ylabel('Successful cases out of 8');ax.set_xlabel('Which vision layer was edited?');ax.set_title('Many layers work: every layer from 10 through 29 succeeds in all eight cases',fontsize=15)
    a=fig.add_subplot(grid[2,:2]);g=sorted({tuple(r['layer_set']) for r in trials if len(r['layer_set'])>1});count=[sum(r['target_transferred'] and r['neighbor_preserved'] for r in trials if tuple(r['layer_set'])==x) for x in g]
    a.bar([f'{x[0]}–{x[-1]}' for x in g],count,color='#6585b5');a.set_ylim(0,9);a.set_ylabel('Successful cases out of 8');a.set_xlabel('Layers edited together');a.set_title('Groups: more edits do not guarantee improvement')
    b=fig.add_subplot(grid[2,2:]);late=[single[28],single[29],single[30],single[31],count[-1]]
    b.bar(['28','29','30','31','28–31\ntogether'],late,color=['#31866b','#31866b','#d58d62','#d58d62','#6585b5']);b.set_ylim(0,9);b.set_ylabel('Successful cases out of 8');b.set_title('The final group works—but layers 28 and 29 already do')
    for i,v in enumerate(late):b.text(i,v+.15,f'{v}/8',ha='center')
    fig.suptitle('Did the target borrow the donor’s color while its neighbor stayed unchanged?',fontsize=18)
    fig.text(.05,.013,'Eight cases reuse four layouts (including reversed donor/recipient pairs), not eight independent scenes.\nThis measures reported color transfer; it does not establish shape binding or a unique color layer.',fontsize=11)
    fig.tight_layout(rect=[0,.06,1,.96]);fig.savefig(out/'eli5-layer-results.png',dpi=160);plt.close(fig)
    body=['<!doctype html><meta charset="utf-8"><title>Vision sweep explained</title><style>body{font:18px system-ui;max-width:1400px;margin:30px auto;padding:20px}img{width:100%}</style><h1>Vision sweep: pictures first</h1>']
    for name in ['eli5-how-it-works.png','eli5-layer-results.png']:
        body.append('<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode((out/name).read_bytes()).decode()+'">')
    body.append('<p>Original screenshots remain unchanged. We copy internal states, then ask separately about the left and right objects. The first figure shows three measured examples, not three sequential processing stages in its bottom row.</p><p>Random and background controls: 0/320 target transfers each. Exact self-patch checks: 320/320 pass. Other-object interventions reproduce the opposite target intervention and are not independent replications.</p>')
    (out/'explained.html').write_text('\n'.join(body))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--dataset',required=True);a=p.parse_args();illustrate(a.run,a.dataset)
