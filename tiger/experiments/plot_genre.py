"""Figure: (A) ML-1M training-data genre distribution (is the catalog diverse?)
and (B) recommendation genre tilt when only gender is flipped (Female% - Male%)."""
import json, os, sys
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(_PKG, '..', 'data', 'ml-1m')
BLUE, ORANGE = '#2a78d6', '#eb6834'   # validated categorical slots 1 & 2 (CVD ΔE 24.7)
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'

# --- (A) training-data genre distribution (interaction-weighted) -------------
item_map = json.load(open(f'{DATA}/item_id_map.json')); new2orig = {v:int(k) for k,v in item_map.items()}
mv = pd.read_csv(f'{DATA}/movies.dat', sep='::', engine='python', header=None, encoding='latin-1',
                 names=['mid','title','genres']).set_index('mid')
inter = json.load(open(f'{DATA}/inter.json'))
tw = Counter()
for seq in inter.values():
    for it0 in seq:
        o = new2orig.get(it0)
        if o in mv.index:
            for g in mv.loc[o,'genres'].split('|'): tw[g]+=1
tot = sum(tw.values())
train = {g: 100*c/tot for g,c in tw.items()}

# --- (B) male vs female recommendation genre share (measured: 300 users, top-10) ---
male = dict(Comedy=17.1,Drama=17.4,Action=11.2,Thriller=10.4,**{'Sci-Fi':6.8},Romance=5.8,Adventure=5.3,
            **{"Children's":4.3},Crime=4.3,Horror=3.6,Animation=2.5,Musical=2.4,War=2.5,Mystery=2.0,
            Fantasy=1.6,Western=1.4,**{'Film-Noir':0.9},Documentary=0.4)
female = dict(Comedy=18.4,Drama=17.9,Action=9.8,Thriller=9.6,**{'Sci-Fi':6.5},Romance=7.0,Adventure=5.4,
              **{"Children's":4.6},Crime=3.8,Horror=3.1,Animation=2.7,Musical=2.6,War=2.2,Mystery=2.1,
              Fantasy=1.8,Western=1.4,**{'Film-Noir':1.0},Documentary=0.3)
delta = {g: female[g]-male[g] for g in male}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 7))
fig.patch.set_facecolor('#fcfcfb')
for ax in (axA, axB):
    ax.set_facecolor('#fcfcfb')
    for s in ('top','right','left'): ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(length=0, colors=MUTED)

# Panel A: sorted descending, single hue
gA = sorted(train, key=train.get)
yA = range(len(gA))
axA.barh(list(yA), [train[g] for g in gA], color=BLUE, height=0.68)
axA.set_yticks(list(yA)); axA.set_yticklabels(gA, color=INK, fontsize=10)
axA.xaxis.grid(True, color=GRID, linewidth=0.8); axA.set_axisbelow(True)
for i,g in enumerate(gA):
    axA.text(train[g]+0.2, i, f'{train[g]:.1f}%', va='center', color=MUTED, fontsize=8.5)
axA.set_xlim(0, max(train.values())*1.16)
axA.set_title('A.  Training-data genre mix — 18 genres (not mono-genre)',
              color=INK, fontsize=12, fontweight='bold', loc='left', pad=12)
axA.set_xlabel('% of training interactions', color=MUTED, fontsize=9.5)

# Panel B: diverging by sign, sorted by delta
gB = sorted(delta, key=delta.get)
yB = range(len(gB))
colors = [ORANGE if delta[g] > 0 else BLUE for g in gB]
axB.barh(list(yB), [delta[g] for g in gB], color=colors, height=0.68)
axB.axvline(0, color=MUTED, linewidth=1)
axB.set_yticks(list(yB)); axB.set_yticklabels(gB, color=INK, fontsize=10)
axB.xaxis.grid(True, color=GRID, linewidth=0.8); axB.set_axisbelow(True)
for i,g in enumerate(gB):
    d = delta[g]
    axB.text(d + (0.04 if d>=0 else -0.04), i, f'{d:+.1f}', va='center',
             ha='left' if d>=0 else 'right', color=MUTED, fontsize=8.5)
axB.set_xlim(-1.8, 1.8)
axB.set_title('B.  Genre shift when gender flipped (Female% − Male%)',
              color=INK, fontsize=12, fontweight='bold', loc='left', pad=12)
axB.set_xlabel('percentage-point change in top-10 genre share', color=MUTED, fontsize=9.5)
axB.legend(handles=[Patch(color=ORANGE, label='more when Female'),
                    Patch(color=BLUE, label='more when Male')],
           loc='lower right', frameon=False, fontsize=9.5, labelcolor=INK)

fig.suptitle('ML-1M is genre-diverse, yet recommendations tilt by gender in a stereotype-consistent way',
             color=INK, fontsize=13.5, fontweight='bold', x=0.02, ha='left', y=0.985)
fig.text(0.02, 0.005, 'Panel B: 300 users, same watch history, only the gender attribute changed · model NDCG@20≈0.167',
         color=MUTED, fontsize=8.5, ha='left')
fig.tight_layout(rect=[0, 0.02, 1, 0.95])
OUT=os.path.join(_PKG, 'experiments', 'genre_distribution.png')
fig.savefig(OUT, dpi=150, facecolor='#fcfcfb')
print('saved', OUT)
