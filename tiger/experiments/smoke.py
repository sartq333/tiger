import json, time, sys
from functools import partial
import torch
from torch.utils.data import DataLoader

# Run from any directory: put the package dir (tiger/tiger) on the path and cd in.
import os
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG)
os.chdir(_PKG)

from modeling import utils
from modeling.dataloader import BatchProcessor
from modeling.dataset import Dataset
from modeling.models import TigerModel, CorrectItemsLogitsProcessor
from modeling.metric import NDCGSemanticMetric, RecallSemanticMetric
from modeling.utils import fix_random_seed

fix_random_seed(42)
config = json.load(open('./configs/tiger_ml1m_config.json'))
print('DEVICE', utils.DEVICE)

t0 = time.time()
dataset = Dataset.create(
    inter_json_path=config['dataset']['inter_json_path'],
    max_sequence_length=config['dataset']['max_sequence_length'],
    sampler_type=config['dataset']['sampler_type'],
    is_extended=True,
)
train_sampler, val_sampler, test_sampler = dataset.get_samplers()
print(f'dataset built in {time.time()-t0:.1f}s | num_items={dataset.num_items}')

num_codebooks = config['dataset']['num_codebooks']
uic = config['model']['user_ids_count']
bp = BatchProcessor.create(config['dataset']['index_json_path'], num_codebooks, uic,
                           user_embeddings_path=config['dataset'].get('user_embeddings_path'))

train_dl = DataLoader(train_sampler, batch_size=config['dataloader']['train_batch_size'],
                      drop_last=True, shuffle=True, collate_fn=bp)
val_dl = DataLoader(val_sampler, batch_size=config['dataloader']['validation_batch_size'],
                    drop_last=False, shuffle=False, collate_fn=bp)

model = TigerModel(
    embedding_dim=config['model']['embedding_dim'], codebook_size=config['model']['codebook_size'],
    sem_id_len=num_codebooks, user_ids_count=uic, num_positions=config['model']['num_positions'],
    num_heads=config['model']['num_heads'], num_encoder_layers=config['model']['num_encoder_layers'],
    num_decoder_layers=config['model']['num_decoder_layers'], dim_feedforward=config['model']['dim_feedforward'],
    num_beams=config['model']['num_beams'], num_return_sequences=config['model']['top_k'],
    activation=config['model']['activation'], d_kv=config['model']['d_kv'], dropout=config['model']['dropout'],
    layer_norm_eps=config['model']['layer_norm_eps'], initializer_range=config['model']['initializer_range'],
    logits_processor=partial(CorrectItemsLogitsProcessor, num_codebooks, config['model']['codebook_size'],
                             config['dataset']['index_json_path'], config['model']['num_beams']),
    user_attr_dim=config['model'].get('user_attr_dim'),
).to(utils.DEVICE)
tp = sum(p.numel() for p in model.parameters())
print(f'model params: {tp:,}  | has user_projection: {hasattr(model,"user_projection")}')

# --- one training step ---
batch = next(iter(train_dl))
print('train batch keys:', sorted(batch.keys()))
print('has user_attr.embedding:', 'user_attr.embedding' in batch)
for k in batch: batch[k] = batch[k].to(utils.DEVICE)
model.train()
t0 = time.time()
out = model(batch)
loss = out['loss']
loss.backward()
print(f'train step OK | loss={loss.item():.4f} | {time.time()-t0:.2f}s')

# --- one validation beam-search batch (times the expensive path) ---
codebook_size = config['model']['codebook_size']
metrics = {'ndcg@10': NDCGSemanticMetric(10, codebook_size, num_codebooks),
           'recall@10': RecallSemanticMetric(10, codebook_size, num_codebooks)}
model.eval()
vb = next(iter(val_dl))
for k in vb: vb[k] = vb[k].to(utils.DEVICE)
t0 = time.time()
with torch.inference_mode():
    vb.update(model(vb))
    r = {m: float(sum(fn(inputs=vb, pred_prefix='predictions', labels_prefix='labels'))/len(fn(inputs=vb, pred_prefix='predictions', labels_prefix='labels'))) for m,fn in metrics.items()}
dt = time.time()-t0
nb = len(val_dl)
print(f'ONE val batch (256 users, 100 beams) generate+metric: {dt:.2f}s')
print(f'val batches total: {nb} -> est full validation pass ~{dt*nb:.0f}s')
print('sample metrics on 1 batch:', r)
print('SMOKE OK')
