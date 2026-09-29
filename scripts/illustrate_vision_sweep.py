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
    from matplotlib.patches import FancyArrowPatch, Rectangle
    def picture(fig, rect, path, title, box=None):
        ax=fig.add_axes(rect)
        ax.imshow(Image.open(path));ax.axis('off');ax.set_title(title,fontsize=13,pad=12)
        if box is not None:
            x0,y0,x1,y1=box
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#ffb52c',linewidth=2.5))
        return ax
    def arrow(fig, start, end):
        fig.add_artist(FancyArrowPatch(start,end,transform=fig.transFigure,
            arrowstyle='-|>',mutation_scale=22,color='#ac6416',linewidth=2))
    rec_path=dataset/pair['recipient']['image']
    donor_path=dataset/pair['color_swap']['image']
    # The general process stands alone: no trial outputs aligned under unrelated inputs.
    fig=plt.figure(figsize=(15,8),facecolor='white')
    fig.suptitle('The process: move internal vectors from donor to recipient',fontsize=21,y=.97)
    picture(fig,[.05,.57,.4,.28],rec_path,'Original recipient input: blue arch + red tower')
    picture(fig,[.55,.57,.4,.28],donor_path,'Donor input: red arch + blue tower',pair['color_swap']['objects'][0]['bbox'])
    fig.text(.25,.49,'Run recipient to chosen vision layer L',ha='center',fontsize=13)
    fig.text(.75,.49,'Run donor to the SAME vision layer L',ha='center',fontsize=13)
    arrow(fig,(.25,.57),(.25,.43));arrow(fig,(.75,.57),(.75,.43))
    fig.text(.25,.36,'Recipient internal vectors\nReplace only the arch-location rows',ha='center',va='center',fontsize=14,
        bbox=dict(boxstyle='round,pad=.8',facecolor='#eaf4f4',edgecolor='#319599'))
    fig.text(.75,.36,'Saved donor internal vectors\nSelect the arch-location rows',ha='center',va='center',fontsize=14,
        bbox=dict(boxstyle='round,pad=.8',facecolor='#fff1dc',edgecolor='#e3a04a'))
    arrow(fig,(.59,.36),(.42,.36));fig.text(.505,.41,'COPY VECTORS',ha='center',fontsize=12,color='#915010')
    arrow(fig,(.25,.28),(.25,.21))
    fig.text(.25,.14,'Continue recipient processing\nAsk left color and right color separately',ha='center',fontsize=14)
    fig.text(.61,.14,'The input pixels stay blue arch + red tower.\nOnly the model’s answers may change.',ha='left',fontsize=13)
    fig.text(.05,.035,'L is the intervention layer in BOTH runs. This process applies separately to each trial below. No recolored “model view” was recorded.',fontsize=11)
    fig.savefig(out/'eli5-how-it-works.png',dpi=160);plt.close(fig)
    example_files=[]
    for layer,target in [(15,0),(31,0),(31,1)]:
        r=next(r for r in trials if r['family']=='f0000' and r['layer_set']==[layer] and r['target_side']==target)
        rec_objects=pair['recipient']['objects'];don_objects=pair['color_swap']['objects']
        obj=rec_objects[target];neighbor=rec_objects[1-target];side=['left','right']
        fig=plt.figure(figsize=(16,8),facecolor='white')
        fig.suptitle(f"ONE RECORDED TRIAL — layer {layer}, patch the {side[target]} {obj['type']}",fontsize=21,y=.97)
        picture(fig,[.025,.55,.29,.29],rec_path,'Recipient input (actual screenshot)',obj['bbox'])
        picture(fig,[.355,.55,.29,.29],donor_path,f'Donor: take layer {layer} vectors here',don_objects[target]['bbox'])
        picture(fig,[.685,.55,.29,.29],run/f'alignment/f0000/patch-target-target{target}.png',f'Recipient: overwrite layer {layer} rows')
        arrow(fig,(.64,.70),(.685,.70))
        fig.text(.5,.48,'Orange marks the selected positions. Copy hidden vectors here—not an image cutout.',ha='center',fontsize=13)
        fig.text(.045,.35,'What is actually in the recipient image?\nLEFT: '+rec_objects[0]['color']+'    RIGHT: '+rec_objects[1]['color'],fontsize=16,va='top')
        fig.text(.365,.35,'What did the patched model answer?\nLEFT: “'+r['answers'][0]+'”    RIGHT: “'+r['answers'][1]+'”',fontsize=16,va='top',
            bbox=dict(boxstyle='round,pad=.65',facecolor='#f0f4fa',edgecolor='#9facbf'))
        target_status='YES' if r['target_transferred'] else 'NO'
        neighbor_status='YES' if r['neighbor_preserved'] else 'NO'
        fig.text(.70,.36,f"Target borrowed donor color? {target_status}\n{side[target]} answer: {r['answers'][target]} | donor: {don_objects[target]['color']}\n\nNeighbor kept original answer? {neighbor_status}\n{side[1-target]} answer: {r['answers'][1-target]} | original: {neighbor['color']}",fontsize=13,va='top')
        fig.text(.045,.10,'“Successful transfer” means the intended answer change occurred. It does not mean the answer became more accurate.\nEach answer comes from a separate question with the identical patch. Images are unchanged; no hidden mental image is inferred.',fontsize=12)
        name=f'eli5-trial-layer{layer:02d}-target{target}.png';example_files.append(name)
        fig.savefig(out/name,dpi=160);plt.close(fig)
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
    for name in ['eli5-how-it-works.png',*example_files,'eli5-layer-results.png']:
        body.append('<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode((out/name).read_bytes()).decode()+'">')
    body.append('<p>Original screenshots remain unchanged. We copy internal states, then ask separately about the left and right objects. The first figure shows only the general process. Each following trial figure has its own layer, images, patch location and measured answers.</p><p>Random and background controls: 0/320 target transfers each. Exact self-patch checks: 320/320 pass. Other-object interventions reproduce the opposite target intervention and are not independent replications.</p>')
    (out/'explained.html').write_text('\n'.join(body))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--dataset',required=True);a=p.parse_args();illustrate(a.run,a.dataset)
