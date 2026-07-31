"""Full ranking-metric snapshot (ndcg/recall @5,10,20) on a random sample of the
ML-1M test set, using the best checkpoint. CPU so it doesn't touch the GPU run."""
import json, argparse
from functools import partial
import numpy as np
import torch

# Run from any directory: put the package dir (tiger/tiger) on the path and cd in.
import os, sys
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG)
os.chdir(_PKG)

_pre = argparse.ArgumentParser(add_help=False); _pre.add_argument('--device', default='cpu')
_a, _ = _pre.parse_known_args()
from modeling import utils
utils.DEVICE = torch.device(_a.device)
from torch.utils.data import DataLoader, Subset
from modeling.dataloader import BatchProcessor
from modeling.dataset import Dataset
from modeling.models import TigerModel, CorrectItemsLogitsProcessor
from modeling.metric import NDCGSemanticMetric, RecallSemanticMetric
from modeling.utils import fix_random_seed

ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True)
ap.add_argument('--n', type=int, default=800); ap.add_argument('--device', default='cpu')
args = ap.parse_args()
fix_random_seed(42)
cfg = json.load(open('./configs/tiger_ml1m_config.json'))
nc = cfg['dataset']['num_codebooks']; cb = cfg['model']['codebook_size']; DEV = utils.DEVICE

ds = Dataset.create(inter_json_path=cfg['dataset']['inter_json_path'],
                    max_sequence_length=cfg['dataset']['max_sequence_length'],
                    sampler_type=cfg['dataset']['sampler_type'], is_extended=True)
_, _, test_sampler = ds.get_samplers()
rng = np.random.default_rng(0)
idx = sorted(rng.choice(len(test_sampler), size=min(args.n, len(test_sampler)), replace=False).tolist())
bp = BatchProcessor.create(cfg['dataset']['index_json_path'], nc, cfg['model']['user_ids_count'],
                           user_embeddings_path=cfg['dataset'].get('user_embeddings_path'))
dl = DataLoader(Subset(test_sampler, idx), batch_size=32, shuffle=False, collate_fn=bp)

model = TigerModel(embedding_dim=cfg['model']['embedding_dim'], codebook_size=cb, sem_id_len=nc,
    user_ids_count=cfg['model']['user_ids_count'], num_positions=cfg['model']['num_positions'],
    num_heads=cfg['model']['num_heads'], num_encoder_layers=cfg['model']['num_encoder_layers'],
    num_decoder_layers=cfg['model']['num_decoder_layers'], dim_feedforward=cfg['model']['dim_feedforward'],
    num_beams=cfg['model']['num_beams'], num_return_sequences=cfg['model']['top_k'],
    activation=cfg['model']['activation'], d_kv=cfg['model']['d_kv'], dropout=cfg['model']['dropout'],
    layer_norm_eps=cfg['model']['layer_norm_eps'], initializer_range=cfg['model']['initializer_range'],
    logits_processor=partial(CorrectItemsLogitsProcessor, nc, cb, cfg['dataset']['index_json_path'],
                             cfg['model']['num_beams']),
    user_attr_dim=cfg['model'].get('user_attr_dim')).to(DEV)
model.load_state_dict(torch.load(args.ckpt, map_location=DEV)); model.eval()

metrics = {f'ndcg@{k}': NDCGSemanticMetric(k, cb, nc) for k in (5,10,20)}
metrics.update({f'recall@{k}': RecallSemanticMetric(k, cb, nc) for k in (5,10,20)})
acc = {m: [] for m in metrics}
with torch.inference_mode():
    for bi, batch in enumerate(dl):
        for k in batch: batch[k] = batch[k].to(DEV)
        batch.update(model(batch))
        for m, fn in metrics.items():
            acc[m].extend(fn(inputs=batch, pred_prefix='predictions', labels_prefix='labels'))
        print(f'  batch {bi+1}/{len(dl)} done', flush=True)
print(f'\n=== TEST metrics on {len(idx)} sampled users (ckpt={args.ckpt.split("/")[-1]}) ===')
for m in ('ndcg@5','ndcg@10','ndcg@20','recall@5','recall@10','recall@20'):
    print(f'  {m:<10} {np.mean(acc[m]):.5f}')
