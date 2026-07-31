"""Gender-gap probe: hold a user's watch history fixed, swap only the gender
attribute in their conditioning embedding, and compare the model's top-K movie
recommendations. Because user embeddings are an exact deterministic function of
"Gender: X. Age: Y. Occupation: Z.", the counterfactual (opposite-gender) vector
is obtained by table lookup from a real user with identical (age, occupation)."""
import json, pickle, sys, argparse
from functools import partial
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

# Make the script runnable from any directory: add the package dir (tiger/tiger,
# the parent of experiments/) to sys.path and cd into it so `import modeling` works
# and ./configs, ../data, ../checkpoints resolve as they do from tiger/tiger.
import os
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG)
os.chdir(_PKG)

# Parse args and pin the device BEFORE importing model modules: tiger.py binds
# `from modeling.utils import DEVICE` at import time (logits processor tensors),
# so utils.DEVICE must be set first or CPU/CUDA will mismatch during generate.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--device', default='cpu')
_dev_args, _ = _pre.parse_known_args()

from modeling import utils
utils.DEVICE = torch.device(_dev_args.device)
from modeling.dataloader import BatchProcessor
from modeling.dataset import Dataset
from modeling.models import TigerModel, CorrectItemsLogitsProcessor
from modeling.utils import fix_random_seed

DATA = '../data/ml-1m'
AGE_MAP = {1:"Under 18",18:"18-24",25:"25-34",35:"35-44",45:"45-49",50:"50-55",56:"56+"}
OCC_MAP = {0:"other",1:"academic/educator",2:"artist",3:"clerical/admin",4:"college/grad student",
           5:"customer service",6:"doctor/health care",7:"executive/managerial",8:"farmer",
           9:"homemaker",10:"K-12 student",11:"lawyer",12:"programmer",13:"retired",
           14:"sales/marketing",15:"scientist",16:"self-employed",17:"technician/engineer",
           18:"tradesman/craftsman",19:"unemployed",20:"writer"}

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='../checkpoints/tiger_ml1m_best.pth')
ap.add_argument('--users', default='')  # comma-sep user ids; empty -> auto-pick
ap.add_argument('--agg', type=int, default=0)  # if >0, also run aggregate genre-shift over N random users
ap.add_argument('--device', default='cpu')
ap.add_argument('--scan', type=int, default=0)      # scan N users, show the most gender-divergent ones
ap.add_argument('--n-strong', type=int, default=4)  # how many strong examples to show
args = ap.parse_args()

fix_random_seed(42)
config = json.load(open('./configs/tiger_ml1m_config.json'))
num_cb = config['dataset']['num_codebooks']; cb = config['model']['codebook_size']
DEV = utils.DEVICE

# ---- item decode maps --------------------------------------------------------
index = json.load(open(f'{DATA}/index_rqvae.json'))                 # 0-idx item -> [4 codes]
code2item = {tuple(v): int(k) for k, v in index.items()}
item_map = json.load(open(f'{DATA}/item_id_map.json'))              # orig movie id -> 0-idx
new2orig = {v: int(k) for k, v in item_map.items()}
mv = pd.read_csv(f'{DATA}/movies.dat', sep='::', engine='python', header=None,
                 encoding='latin-1', names=['mid','title','genres'])
mv = mv.set_index('mid')
def item_title(it0):
    o = new2orig.get(it0);
    return (mv.loc[o,'title'], mv.loc[o,'genres']) if o in mv.index else (f'item{it0}', '')

def decode(pred_row):  # (20,4) offset-space -> list of (item0, title, genres)
    out = []
    offs = cb * np.arange(num_cb)
    for r in pred_row.tolist():
        codes = tuple(int(c - o) for c, o in zip(r, offs))
        it = code2item.get(codes)
        if it is None:
            out.append((None, '<invalid>', '')); continue
        t, g = item_title(it); out.append((it, t, g))
    return out

# ---- user demographic / embedding lookups -----------------------------------
d = pickle.load(open(f'{DATA}/user_embeddings.pkl','rb'))
uemb = {int(u): np.array(e, dtype=np.float32) for u, e in zip(d['item_id'], d['embedding'])}
udf = pd.read_csv(f'{DATA}/users.dat', sep='::', engine='python', header=None, encoding='latin-1',
                  names=['user_id','gender','age','occupation','zip'])
u_demo = {int(r.user_id): (r.gender, int(r.age), int(r.occupation)) for _, r in udf.iterrows()}
demo2emb = {}                                       # (gender,age,occ) -> embedding
for uid,(g,a,o) in u_demo.items(): demo2emb.setdefault((g,a,o), uemb[uid])
def counterfactual_emb(uid):
    g,a,o = u_demo[uid]; fg = 'F' if g=='M' else 'M'
    return demo2emb.get((fg,a,o)), fg
def demo_str(g,a,o):
    return f"Gender: {'Male' if g=='M' else 'Female'}. Age: {AGE_MAP.get(a,a)}. Occupation: {OCC_MAP.get(o,'?')}."

# ---- dataset / model ---------------------------------------------------------
dataset = Dataset.create(inter_json_path=config['dataset']['inter_json_path'],
                         max_sequence_length=config['dataset']['max_sequence_length'],
                         sampler_type=config['dataset']['sampler_type'], is_extended=True)
_, _, test_sampler = dataset.get_samplers()
uid2idx = {int(s['user.ids'][0]): i for i, s in enumerate(test_sampler.dataset)}
bp = BatchProcessor.create(config['dataset']['index_json_path'], num_cb, config['model']['user_ids_count'],
                           user_embeddings_path=config['dataset'].get('user_embeddings_path'))

model = TigerModel(embedding_dim=config['model']['embedding_dim'], codebook_size=cb, sem_id_len=num_cb,
    user_ids_count=config['model']['user_ids_count'], num_positions=config['model']['num_positions'],
    num_heads=config['model']['num_heads'], num_encoder_layers=config['model']['num_encoder_layers'],
    num_decoder_layers=config['model']['num_decoder_layers'], dim_feedforward=config['model']['dim_feedforward'],
    num_beams=config['model']['num_beams'], num_return_sequences=config['model']['top_k'],
    activation=config['model']['activation'], d_kv=config['model']['d_kv'], dropout=config['model']['dropout'],
    layer_norm_eps=config['model']['layer_norm_eps'], initializer_range=config['model']['initializer_range'],
    logits_processor=partial(CorrectItemsLogitsProcessor, num_cb, cb, config['dataset']['index_json_path'],
                             config['model']['num_beams']),
    user_attr_dim=config['model'].get('user_attr_dim')).to(DEV)
model.load_state_dict(torch.load(args.ckpt, map_location=DEV)); model.eval()
print(f'Loaded checkpoint: {args.ckpt}')

def recommend(uids, emb_override=None):
    """emb_override: optional (len(uids), 4096) tensor replacing user_attr.embedding."""
    batch = bp([test_sampler[uid2idx[u]] for u in uids])
    if emb_override is not None:
        batch['user_attr.embedding'] = emb_override
    for k in batch: batch[k] = batch[k].to(DEV)
    with torch.inference_mode():
        out = model(batch)
    return out['predictions'].cpu().numpy()   # (B,20,4)

def pick_users(n=3):
    # users whose (age,occ) has both genders, decent-length history, varied demographics
    cand = [u for u,(g,a,o) in u_demo.items()
            if ('M',a,o) in demo2emb and ('F',a,o) in demo2emb and u in uid2idx]
    cand.sort()
    seen_demo, out = set(), []
    for u in cand:
        g,a,o = u_demo[u]
        if (a,o) in seen_demo: continue
        seen_demo.add((a,o)); out.append(u)
        if len(out) >= n: break
    return out

def genre_counts(recs):
    c = defaultdict(int)
    for it,t,g in recs:
        for gg in (g.split('|') if g else []): c[gg]+=1
    return c

def emb_for(demo):
    return torch.tensor(np.stack([demo2emb[demo]]), dtype=torch.float32)

def hist_titles(u):
    seq = test_sampler[uid2idx[u]]['item.ids']
    return [item_title(it)[0] for it in seq]

TOPN = 10

# Strong, validated examples: users (with both genders available) where flipping
# ONLY the gender attribute changes the *genres* of the top-10 the most. Found by
# scanning with --scan and ranking by genre-distribution L1 distance, then baked
# in so a plain `python gender_gap.py` run surfaces clear bias out of the box.
# Found via `--scan 500` on tiger_ml1m_best.pth (NDCG@20~0.167), ranked by single-user
# genre-L1. Re-run --scan after retraining, since the strongest users are checkpoint-specific.
#   3713 female-exec: Male->Fight Club/Unforgiven/Misery ; Female->Bambi/Fantasia/Casablanca (L1=18)
#   1080 56+ male   : Male->Conan/Star Trek/Logan's Run  ; Female->You've Got Mail/rom-coms  (L1=17)
#   4509 male exec  : Male->Robocop/Devil's Advocate     ; Female->Ghost/Fried Green Tomatoes(L1=11)
#   4739 male admin : Male->Godfather/Jurassic Park      ; Female->Scary Movie/Best in Show   (L1=11)
DEFAULT_STRONG_USERS = [3713, 1080, 4509, 4739]

def cell(rec):  # "Title [Genre/Genre]" for the side-by-side view
    _, t, gg = rec
    prim = '/'.join(gg.split('|')[:2]) if gg else '?'
    return f'{t[:23]} [{prim}]'[:41]

def recs_mf(u):  # top-K decoded recs conditioning the same history as Male / as Female
    a_o = u_demo[u][1:]
    return (decode(recommend([u], emb_override=emb_for(('M',) + a_o))[0]),
            decode(recommend([u], emb_override=emb_for(('F',) + a_o))[0]))

def genre_l1(male, female):  # L1 distance between the two top-K genre distributions
    gm, gf = genre_counts(male[:TOPN]), genre_counts(female[:TOPN])
    return sum(abs(gm[k] - gf[k]) for k in set(gm) | set(gf))

def find_strong_users(k):
    # NOTE: beam-search generation is batch-dependent (FP non-associativity across
    # batch size, amplified by the 100-beam constrained decoder), so a user's recs
    # in a big batch differ from its single-user recs. We therefore shortlist cheaply
    # with a batched pass, then RE-SCORE the shortlist one user at a time so the chosen
    # examples reproduce exactly in the single-user display below.
    rng = np.random.default_rng(0)
    pool = [u for u,(g,a,o) in u_demo.items()
            if ('M',a,o) in demo2emb and ('F',a,o) in demo2emb and u in uid2idx]
    pool = list(map(int, rng.choice(pool, size=min(args.scan, len(pool)), replace=False)))
    male_e = np.stack([demo2emb[('M',)+u_demo[u][1:]] for u in pool])
    fem_e  = np.stack([demo2emb[('F',)+u_demo[u][1:]] for u in pool])
    shortlist, B = [], 128
    for i in range(0, len(pool), B):
        us = pool[i:i+B]
        rm = recommend(us, torch.tensor(male_e[i:i+B], dtype=torch.float32))
        rf = recommend(us, torch.tensor(fem_e[i:i+B], dtype=torch.float32))
        for j, u in enumerate(us):
            shortlist.append((genre_l1(decode(rm[j]), decode(rf[j])), u))
    shortlist = [u for _, u in sorted(shortlist, reverse=True)[:max(4*k, 20)]]
    scored = sorted(((genre_l1(*recs_mf(u)), u) for u in shortlist), reverse=True)  # single-user
    print(f'\nMost gender-divergent of {len(pool)} scanned (single-user genre-L1): '
          + ', '.join(f'{u}(L1={s})' for s, u in scored[:k]))
    return [u for _, u in scored[:k]]

if args.users:
    sample_users = [int(x) for x in args.users.split(',')]
elif args.scan > 0:
    sample_users = find_strong_users(args.n_strong)
else:
    sample_users = DEFAULT_STRONG_USERS

print('\n=== MALE vs FEMALE recommendations (same history, only gender changed) ===')
for u in sample_users:
    g, a, o = u_demo[u]
    male, female = recs_mf(u)
    tm = [t for _, t, _ in male]; tf = [t for _, t, _ in female]
    common = set(tm[:TOPN]) & set(tf[:TOPN])
    print(f'\n{"="*90}')
    print(f'USER {u}  |  Age: {AGE_MAP.get(a,a)}  Occupation: {OCC_MAP.get(o,"?")}  (actual: {"Male" if g=="M" else "Female"})')
    print(f'recent history: {" | ".join(hist_titles(u)[-5:])}')
    print(f'{"-"*90}')
    print(f'{"#":>2}  {"conditioned MALE":<42} {"conditioned FEMALE":<42}')
    for i in range(TOPN):
        mc = cell(male[i]) if i < len(male) else ''
        fc = cell(female[i]) if i < len(female) else ''
        mk = ' ' if tm[i] in common else '*'
        fk = ' ' if tf[i] in common else '*'
        print(f'{i+1:>2}  {mk}{mc:<42} {fk}{fc:<42}')
    print(f'{"-"*90}')
    gm = genre_counts(male[:TOPN]); gf = genre_counts(female[:TOPN])
    fmt = lambda c: ', '.join(f'{k}x{v}' for k, v in sorted(c.items(), key=lambda x:-x[1]))
    print(f'genre mix  MALE  : {fmt(gm)}')
    print(f'genre mix  FEMALE: {fmt(gf)}')
    print(f'top-{TOPN} title overlap: {len(common)}/{TOPN}   genre-L1 distance: {genre_l1(male, female)}   (* = only that gender)')

# ---- optional aggregate genre shift -----------------------------------------
if args.agg > 0:
    rng = np.random.default_rng(0)
    pool = [u for u,(g,a,o) in u_demo.items()
            if ('M',a,o) in demo2emb and ('F',a,o) in demo2emb and u in uid2idx]
    pool = list(map(int, rng.choice(pool, size=min(args.agg,len(pool)), replace=False)))
    male_emb = np.stack([demo2emb[('M',)+u_demo[u][1:]] for u in pool])
    fem_emb  = np.stack([demo2emb[('F',)+u_demo[u][1:]] for u in pool])
    gm, gf = defaultdict(int), defaultdict(int)
    overlap = []
    B = 128
    for i in range(0, len(pool), B):
        us = pool[i:i+B]
        rm = recommend(us, torch.tensor(male_emb[i:i+B], dtype=torch.float32))
        rf = recommend(us, torch.tensor(fem_emb[i:i+B], dtype=torch.float32))
        for j in range(len(us)):
            dm, dfem = decode(rm[j]), decode(rf[j])
            for c,v in genre_counts(dm).items(): gm[c]+=v
            for c,v in genre_counts(dfem).items(): gf[c]+=v
            sm, sf = {t for _,t,_ in dm}, {t for _,t,_ in dfem}
            overlap.append(len(sm & sf)/max(len(sm | sf),1))
    tot_m = sum(gm.values()) or 1; tot_f = sum(gf.values()) or 1
    print(f'\n=== AGGREGATE over {len(pool)} users (same history, condition as Male vs Female) ===')
    print(f'mean top-20 Jaccard(Male,Female) = {np.mean(overlap):.3f}  (1.0 = gender changes nothing)')
    allg = sorted(set(gm)|set(gf), key=lambda k:-(gm[k]+gf[k]))
    print('genre share %  (Male -> Female, Δpp):')
    for k in allg:
        pm, pf = 100*gm[k]/tot_m, 100*gf[k]/tot_f
        if abs(pf-pm) >= 0.3:
            print(f'    {k:<12} {pm:5.1f} -> {pf:5.1f}  ({pf-pm:+.1f})')

print('\nDONE')
