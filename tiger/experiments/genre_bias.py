"""Genre-level gender analysis: does conditioning as Male vs Female change the
GENRE mix of recommendations (not just specific titles)? Stratifies by whether
the user's watch history is genre-homogeneous or diverse, to test the hypothesis
that bias is masked for strongly-typed users."""
import json, pickle, sys, argparse
from functools import partial
from collections import defaultdict, Counter
import numpy as np, pandas as pd, torch

import os
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG); os.chdir(_PKG)

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='../checkpoints/tiger_ml1m_best.pth')
ap.add_argument('--n', type=int, default=300)
ap.add_argument('--device', default='cuda')
ap.add_argument('--topk', type=int, default=10)
args = ap.parse_args()

import modeling.utils as U; U.DEVICE = torch.device(args.device)
from modeling.dataloader import BatchProcessor
from modeling.dataset import Dataset
from modeling.models import TigerModel, CorrectItemsLogitsProcessor
from modeling.utils import fix_random_seed
fix_random_seed(42)
DATA='../data/ml-1m'; cfg=json.load(open('./configs/tiger_ml1m_config.json'))
nc=cfg['dataset']['num_codebooks']; cb=cfg['model']['codebook_size']; DEV=U.DEVICE; K=args.topk

index=json.load(open(f'{DATA}/index_rqvae.json')); code2item={tuple(v):int(k) for k,v in index.items()}
item_map=json.load(open(f'{DATA}/item_id_map.json')); new2orig={v:int(k) for k,v in item_map.items()}
mv=pd.read_csv(f'{DATA}/movies.dat',sep='::',engine='python',header=None,encoding='latin-1',
               names=['mid','title','genres']).set_index('mid')
def genres_of(it0):
    o=new2orig.get(it0)
    return mv.loc[o,'genres'].split('|') if o in mv.index else []
def decode_genres(pred_row):
    offs=cb*np.arange(nc); out=[]
    for r in pred_row.tolist():
        codes=tuple(int(c-o) for c,o in zip(r,offs)); it=code2item.get(codes)
        out.append(genres_of(it) if it is not None else [])
    return out

d=pickle.load(open(f'{DATA}/user_embeddings.pkl','rb'))
uemb={int(u):np.array(e,dtype=np.float32) for u,e in zip(d['item_id'],d['embedding'])}
udf=pd.read_csv(f'{DATA}/users.dat',sep='::',engine='python',header=None,encoding='latin-1',
                names=['user_id','gender','age','occupation','zip'])
u_demo={int(r.user_id):(r.gender,int(r.age),int(r.occupation)) for _,r in udf.iterrows()}
demo2emb={}
for uid,(g,a,o) in u_demo.items(): demo2emb.setdefault((g,a,o),uemb[uid])

ds=Dataset.create(inter_json_path=cfg['dataset']['inter_json_path'],max_sequence_length=cfg['dataset']['max_sequence_length'],
                  sampler_type=cfg['dataset']['sampler_type'],is_extended=True)
_,_,test_sampler=ds.get_samplers()
uid2idx={int(s['user.ids'][0]):i for i,s in enumerate(test_sampler.dataset)}
bp=BatchProcessor.create(cfg['dataset']['index_json_path'],nc,cfg['model']['user_ids_count'],
                         user_embeddings_path=cfg['dataset'].get('user_embeddings_path'))
model=TigerModel(embedding_dim=cfg['model']['embedding_dim'],codebook_size=cb,sem_id_len=nc,
  user_ids_count=cfg['model']['user_ids_count'],num_positions=cfg['model']['num_positions'],num_heads=cfg['model']['num_heads'],
  num_encoder_layers=cfg['model']['num_encoder_layers'],num_decoder_layers=cfg['model']['num_decoder_layers'],
  dim_feedforward=cfg['model']['dim_feedforward'],num_beams=cfg['model']['num_beams'],num_return_sequences=cfg['model']['top_k'],
  activation=cfg['model']['activation'],d_kv=cfg['model']['d_kv'],dropout=cfg['model']['dropout'],
  layer_norm_eps=cfg['model']['layer_norm_eps'],initializer_range=cfg['model']['initializer_range'],
  logits_processor=partial(CorrectItemsLogitsProcessor,nc,cb,cfg['dataset']['index_json_path'],cfg['model']['num_beams']),
  user_attr_dim=cfg['model'].get('user_attr_dim')).to(DEV)
model.load_state_dict(torch.load(args.ckpt,map_location=DEV)); model.eval()

def recommend(uids, emb):
    batch=bp([test_sampler[uid2idx[u]] for u in uids]); batch['user_attr.embedding']=torch.tensor(emb,dtype=torch.float32)
    for k in batch: batch[k]=batch[k].to(DEV)
    with torch.inference_mode(): out=model(batch)
    return out['predictions'].cpu().numpy()

def hist_genre_homogeneity(u):  # fraction of history in its single top genre
    seq=test_sampler[uid2idx[u]]['item.ids']; c=Counter()
    for it in seq:
        for g in genres_of(it): c[g]+=1
    return (c.most_common(1)[0][1]/sum(c.values())) if c else 0.0

rng=np.random.default_rng(0)
pool=[u for u,(g,a,o) in u_demo.items() if ('M',a,o) in demo2emb and ('F',a,o) in demo2emb and u in uid2idx]
pool=list(map(int,rng.choice(pool,size=min(args.n,len(pool)),replace=False)))
male_e=np.stack([demo2emb[('M',)+u_demo[u][1:]] for u in pool])
fem_e =np.stack([demo2emb[('F',)+u_demo[u][1:]] for u in pool])

gm,gf=defaultdict(int),defaultdict(int)            # genre share over top-K
t1m,t1f=defaultdict(int),defaultdict(int)          # genre of the #1 rec
same_top1=0; title_jac_hom=[]; title_jac_div=[]
B=128
for i in range(0,len(pool),B):
    us=pool[i:i+B]
    rm=recommend(us,male_e[i:i+B]); rf=recommend(us,fem_e[i:i+B])
    offs=cb*np.arange(nc)
    for j,u in enumerate(us):
        gm_rows=decode_genres(rm[j][:K]); gf_rows=decode_genres(rf[j][:K])
        for row in gm_rows:
            for g in row: gm[g]+=1
        for row in gf_rows:
            for g in row: gf[g]+=1
        for g in gm_rows[0]: t1m[g]+=1
        for g in gf_rows[0]: t1f[g]+=1
        # same #1 title?
        def item_of(r):
            codes=tuple(int(c-o) for c,o in zip(r,offs)); return code2item.get(codes)
        if item_of(rm[j][0])==item_of(rf[j][0]): same_top1+=1
        # title jaccard, stratified by history homogeneity
        sm={item_of(r) for r in rm[j][:K]}; sf={item_of(r) for r in rf[j][:K]}
        jac=len(sm&sf)/max(len(sm|sf),1)
        (title_jac_hom if hist_genre_homogeneity(u)>=0.5 else title_jac_div).append(jac)

tm=sum(gm.values()) or 1; tf=sum(gf.values()) or 1
print(f"\n=== GENRE-LEVEL gender analysis over {len(pool)} users (top-{K}, ckpt={args.ckpt.split('/')[-1]}) ===")
print(f"#1 recommendation is the SAME movie for M & F: {same_top1}/{len(pool)} ({100*same_top1/len(pool):.0f}%)")
print(f"\nGenre share in top-{K}  (Male% -> Female%, Δpp):")
for g in sorted(set(gm)|set(gf),key=lambda k:-(gm[k]+gf[k])):
    pm,pf=100*gm[g]/tm,100*gf[g]/tf
    flag=' <--' if abs(pf-pm)>=0.5 else ''
    print(f"   {g:<12} {pm:5.1f} -> {pf:5.1f}  ({pf-pm:+.1f}){flag}")
t1tm=sum(t1m.values()) or 1; t1tf=sum(t1f.values()) or 1
print(f"\nGenre of the #1 rec  (Male% -> Female%):")
for g in sorted(set(t1m)|set(t1f),key=lambda k:-(t1m[k]+t1f[k]))[:8]:
    print(f"   {g:<12} {100*t1m[g]/t1tm:5.1f} -> {100*t1f[g]/t1tf:5.1f}")
print(f"\nGender effect stratified by history type (mean top-{K} title Jaccard; lower = gender matters more):")
print(f"   homogeneous-history users (top genre >=50%): Jaccard {np.mean(title_jac_hom):.3f}  (n={len(title_jac_hom)})")
print(f"   diverse-history users     (top genre < 50%): Jaccard {np.mean(title_jac_div):.3f}  (n={len(title_jac_div)})")
