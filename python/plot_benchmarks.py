"""Render concise benchmark figures from the recorded report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / 'benchmarks/RESULTS.json').read_text())['rows']
lookup = {(r['candidate'], r['opponent']): r for r in rows}
out = ROOT / 'benchmarks/charts'
out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': '#fafaf7', 'axes.facecolor': '#fafaf7'})

pairs = [('rho','iota'), ('rho','beta'), ('iota','beta')]
fig, ax = plt.subplots(figsize=(12, 6.5))
y = np.arange(3)
for offset, key, label, color in [(-.23, None, 'Overall (includes forfeits)', '#267c80'),
                                 (0, 'Attacker', 'As attacker', '#d79537'),
                                 (.23, 'Defender', 'As defender', '#7267a5')]:
    values = [100*(lookup[p]['summary']['score_rate'] if key is None else
                   lookup[p]['summary']['by_side'][key]['score_rate']) for p in pairs]
    ax.barh(y+offset, values, height=.20, color=color, label=label)
    for yy, value in zip(y+offset, values):
        ax.text(value+1.3, yy, f'{value:.0f}%', va='center', fontsize=11)
ax.set_yticks(y, [f'{a.title()} vs {b.title()}\n{lookup[(a,b)]["failures"]}/50 forfeits' for a,b in pairs])
ax.invert_yaxis()
ax.set_xlim(0, 105)
ax.set_xticks([0,25,50,75,100], ['0%','25%','50%','75%','100%'])
ax.axvline(50, color='#999', linewidth=1, linestyle='--', zorder=0)
ax.set_xlabel('First bot’s match score · 25 games per side')
ax.legend(loc='lower center', bbox_to_anchor=(.5, -0.27), ncol=3, frameon=False, fontsize=10)
fig.suptitle('Skallagrim: internal head-to-head', x=.06, ha='left', fontsize=22, weight='bold')
fig.text(.06,.89,'50 games per pairing · 300 ms search + 25 ms IPC allowance', color='#555')
fig.text(.06,.025,'Forfeits count either player. Rho–Iota overall 95% interval: 46–70%; no clear winner established.', fontsize=10, color='#555')
fig.subplots_adjust(left=.21,right=.95,top=.80,bottom=.25)
fig.savefig(out/'trust-me-bro-internal.png', dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1,3,figsize=(12,8),gridspec_kw={'width_ratios':[1,1,1.1]})
ordered = [(c,o) for o in ['luizz','nicholas','opentafl','taflzero'] for c in ['rho','iota','beta']]
labels = {'luizz':'LuizzK','nicholas':'Nicholas','opentafl':'OpenTafl','taflzero':'TaflZero'}
raw = np.array([lookup[p]['summary']['score_rate']*100 for p in ordered])
played = np.array([np.nan if lookup[p]['played_score'] is None else lookup[p]['played_score']*100 for p in ordered])
cmap = plt.get_cmap('YlGnBu').copy()
cmap.set_bad('#e6e6e2')
for ax, values, title in zip(axes[:2], [raw,played], ['Overall score\nincludes forfeits','Completed-play score\nexcludes forfeits']):
    ax.imshow(values[:,None], cmap=cmap, vmin=0,vmax=100,aspect='auto')
    ax.set_title(title, fontsize=12, pad=14)
    ax.set_xticks([])
    ax.set_yticks(np.arange(12))
    for i,v in enumerate(values):
        ax.text(0,i,'—' if np.isnan(v) else f'{v:.1f}%',ha='center',va='center',color='white' if v>=65 else '#222',weight='bold')
    for boundary in [2.5,5.5,8.5]:
        ax.axhline(boundary,color='#fafaf7',linewidth=4)
    for spine in ax.spines.values(): spine.set_visible(False)
axes[0].set_yticklabels([f'{c.title()} vs {labels[o]}' for c,o in ordered],fontsize=11)
axes[1].set_yticklabels([])
ax=axes[2]
ax.set_xlim(0,1); ax.set_ylim(11.5,-.5); ax.axis('off')
ax.set_title('Games completed\nwithout failure',fontsize=12,pad=14)
for i,p in enumerate(ordered):
    r=lookup[p]
    ax.text(.5,i,f'{r["played_games"]}/50',ha='center',va='center',weight='bold',color='#333')
fig.suptitle('Skallagrim vs external engines',x=.06,ha='left',fontsize=22,weight='bold')
fig.text(.06,.91,'50 games per pairing · 300 ms search + 25 ms IPC allowance',color='#555')
fig.text(.06,.055,'OpenTafl and TaflZero: every game ended in a timeout forfeit. No completed-play comparison.',fontsize=10,color='#555')
fig.text(.06,.028,'Python opponents use our warm-start weights and have rule mismatches. Completed-play scores may be selection-biased.',fontsize=9.5,color='#555')
fig.subplots_adjust(left=.24,right=.97,top=.82,bottom=.13,wspace=.13)
fig.savefig(out/'trust-me-bro-external.png',dpi=180)
plt.close(fig)
print(out)
