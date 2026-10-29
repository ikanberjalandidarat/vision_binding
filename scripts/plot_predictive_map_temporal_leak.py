#!/usr/bin/env python3
"""Publication figure from the user's pasted Oscar 8098889 report.

Frame tiles and arrows are explanatory schematics, not Minecraft recordings.
No image differences or uncertainty estimates are fabricated. No model is run.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'output/pdf/spatial-video'
OUT.mkdir(parents=True, exist_ok=True)
DATA = {
    'provenance': 'Transcribed from user-pasted Oscar job 8098889 report.json, 2026-10-29; not a new model run',
    'repo_commit': 'fed80b72925a4d3f3e9ec4b70f71ecaedec75c13',
    'checkpoint_sha256': '356a2a7a51217030b5223e2b4f9f9a9e65d51717e042c03b7549afe205f156b0',
    'original_episode': '1675233473.792766', 'donor_episode': '1675233477.3724594',
    'output_index': 4, 'input_frames': 10,
    'conditions': ['Identical repeat', 'Replace frame 5', 'Replace frame 9'],
    'prediction_mean_absolute_change': [0.0, 0.021320130676031113, 0.021900683641433716],
    'latent_mean_absolute_change': [0.0, 0.059479013085365295, 0.06511174142360687],
    'prediction_max_absolute_change': [0.0, 0.34071195125579834, 0.3020581901073456],
    'latent_max_absolute_change': [0.0, 0.2645838260650635, 0.32364487648010254],
    'attention_heads': {'attention_1': 2, 'attention_2': 2},
    'scope': 'One original clip, one donor episode, two future-frame substitutions; no population uncertainty estimate',
}
(OUT / 'temporal-leakage-measurements.json').write_text(json.dumps(DATA, indent=2)+'\n')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.spines.top': False, 'axes.spines.right': False})
INK='#192B3A'; MUTED='#5D6C79'; BLUE='#0072B2'; ORANGE='#D55E00'; TEAL='#00856A'
PALE='#EAF4FA'; GREY='#EDF0F3'; BORDER='#D9E0E6'
fig=plt.figure(figsize=(16,10.6), facecolor='white')
fig.text(.045,.963,'Future frames change an earlier prediction',fontsize=23,weight='bold',color=INK)
fig.text(.045,.928,'Frozen reference predictive coder  |  original Minecraft validation PNGs  |  Oscar 8098889',fontsize=12,color=MUTED)

def panel(bounds,letter,title):
    ax=fig.add_axes(bounds); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    ax.text(0,1,letter,fontsize=17,weight='bold',color=INK,va='top')
    ax.text(.055,.995,title,fontsize=14,weight='bold',color=INK,va='top')
    return ax

def box(ax,x,y,w,h,text,face=GREY,edge=BORDER,color=INK,size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.013',
                               facecolor=face,edgecolor=edge,lw=1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color=color)

def arrow(ax,a,b,color=MUTED,style='-',rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.5,
                               color=color,linestyle=style,connectionstyle=f'arc3,rad={rad}'))

# A: explicit single-frame counterfactuals.
a=panel([.045,.535,.445,.35],'a','The intervention: change only the future')
x0=.205; step=.076; w=.059
for row,(label,changed) in enumerate([('Original',None),('Swap frame 5',5),('Swap frame 9',9)]):
    y=.72-row*.17
    a.text(0,y+.05,label,va='center',fontsize=10.5,color=INK)
    for t in range(10):
        altered=t==changed
        intervention_color = TEAL if changed == 9 else ORANGE
        face=intervention_color if altered else (PALE if t<=4 else GREY)
        box(a,x0+t*step,y,w,.105,f'{t}'+('*' if altered else ''),face=face,
            edge=intervention_color if altered else BORDER,color='white' if altered else INK,size=10)
cut=x0+4*step+w+.009
a.plot([cut,cut],[.355,.87],color=BLUE,lw=1.4,ls='--')
a.text(x0,.885,'Frames 0–4 fixed',color=BLUE,fontsize=11)
a.text(cut+.03,.885,'Frames 5–9 are future',color=MUTED,fontsize=11)
a.text(.205,.28,'* Same-index frame from a second original episode.',fontsize=10,color=MUTED)
box(a,.02,.09,.94,.13,'Read out the same index each time: prediction ŷ₅ and latent z₄',face=PALE,edge=PALE,size=11)
a.text(.02,.025,'Frozen weights • eval mode • identical repeat as a numerical control',fontsize=10,color=MUTED)

# B: inferred normalization bypass; avoid claiming mechanism isolated.
b=panel([.54,.535,.415,.35],'b','How future information can bypass the mask')
box(b,.02,.66,.35,.16,'First attention output at frame 4\nmasked to frames ≤ 4',face=PALE,edge=PALE,size=10.5)
box(b,.61,.66,.35,.16,'Later-frame features\nframes 5–9',face='#FFF1E8',edge='#FFF1E8',size=10.5)
box(b,.26,.36,.48,.17,'Shared normalization statistics\nmean and variance over T × C × H × W',size=10.5)
arrow(b,(.2,.65),(.36,.54),BLUE); arrow(b,(.79,.65),(.65,.54),ORANGE)
box(b,.27,.105,.46,.12,'Normalized frame-4 feature → output',face=PALE,edge=PALE,size=10.2)
arrow(b,(.5,.35),(.5,.235),ORANGE)
b.text(.5,.285,r'$\widetilde{h}_4=(h_4-\mu_{\mathrm{clip}})/\sqrt{\sigma^2_{\mathrm{clip}}+\epsilon}$',
       ha='center',va='center',fontsize=13,color=INK,bbox=dict(facecolor='white',edgecolor='none',pad=2))
b.text(.02,.035,'Source-supported mechanism; normalization has not been ablated.',fontsize=10,color=MUTED)

# C: paired panels with distinct scale and units.
c=panel([.045,.105,.445,.365],'c','Measured dependence on future frames')
c.text(.055,.875,'Mean absolute change relative to the original input',fontsize=11,color=MUTED)
labels=['Identical\nrepeat','Swap\nframe 5','Swap\nframe 9']
for bounds,key,title,limit,ticks in [
    ([.075,.205,.18,.165],'prediction_mean_absolute_change','Prediction (normalized image units)',.028,[0,.01,.02]),
    ([.305,.205,.18,.165],'latent_mean_absolute_change','Latent (activation units)',.083,[0,.03,.06])]:
    ax=fig.add_axes(bounds); vals=DATA[key]
    ax.bar(np.arange(3),vals,color=[MUTED,ORANGE,TEAL],width=.58,zorder=3)
    ax.scatter([0],[0],marker='o',s=22,color=MUTED,zorder=4,clip_on=False)
    for i,v in enumerate(vals):
        ax.text(i,v+limit*.035,'0' if v==0 else f'{v:.5f}',ha='center',va='bottom',fontsize=10,color=INK)
    ax.set_ylim(0,limit); ax.set_xticks(range(3),labels,fontsize=9.5)
    ax.set_yticks(ticks); ax.tick_params(axis='both',length=0,labelcolor=MUTED,pad=5)
    ax.spines['left'].set_visible(False); ax.spines['bottom'].set_color(BORDER)
    ax.grid(axis='y',color=GREY,zorder=0); ax.set_title(title,fontsize=10.5,color=INK,pad=14)
c.text(.055,.135,'A causal output must remain unchanged when only its future changes.',fontsize=10.5,color=INK)
c.text(.055,.045,'One clip + one donor; descriptive values, no error bars.\nImage-output and latent changes have different units.',fontsize=10,color=MUTED,linespacing=1.5)

# D: implications and planned boundary-safe extraction.
d=panel([.54,.105,.415,.365],'d','What follows — and what does not')
d.text(.02,.84,'Confirmed',color=ORANGE,fontsize=11,weight='bold')
d.text(.02,.775,'The tested frame-4 output depends on future input.\nThe earlier 0.01293 RGB MSE is not a causal forecasting score.',
       fontsize=11,color=INK,va='top',linespacing=1.55)
d.text(.02,.60,'Not established',color=MUTED,fontsize=11,weight='bold')
d.text(.02,.535,'How much leakage improved that MSE; whether spatial maps\nexist; or whether every evaluation in the paper is affected.',
       fontsize=11,color=INK,va='top',linespacing=1.55)
d.text(.02,.36,'Prepared spatial extraction',color=TEAL,fontsize=11,weight='bold')
for t in range(10):
    box(d,.025+t*.075,.20,.058,.09,str(t),face=PALE if t<9 else TEAL,
        edge=PALE if t<9 else TEAL,color=INK if t<9 else 'white',size=9)
d.text(.82,.245,'No later\nframes',fontsize=10,color=MUTED,ha='center',va='center')
d.plot([.775,.775],[.17,.31],color=TEAL,lw=1.5)
d.text(.02,.105,'Keep only z₉ from a window ending at frame 9.\nNo future input at that endpoint; this is not retraining or a full replication.',
       fontsize=10,color=INK,va='top',linespacing=1.55)
fig.add_artist(plt.Line2D([.045,.955],[.078,.078],color=BORDER,lw=1,transform=fig.transFigure))
fig.text(.045,.052,'Evidence: Oscar 8098889, reported 29 Oct 2026. Reference commit fed80b72925a; frozen PredictiveCoder, two heads per attention module.',fontsize=9,color=MUTED)
fig.text(.045,.032,'Frame tiles are schematic. Changes are measured on the same output index after a single future-frame substitution. Model source: jgornet/predictive-coding-recovers-maps.',fontsize=9,color=MUTED)
for suffix in ('pdf','svg','png'):
    fig.savefig(OUT/f'temporal-leakage-audit.{suffix}',dpi=220,facecolor='white',metadata={'Creator':'Minecraft-Binding temporal leakage audit'} if suffix=='pdf' else None)
plt.close(fig)
print(OUT/'temporal-leakage-audit.pdf')
