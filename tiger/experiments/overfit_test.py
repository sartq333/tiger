"""Overfit sanity check: can the attribute-conditioned model memorize a single
tiny batch? If loss does NOT collapse toward ~0, the training pipeline is broken.
Runs on CPU so it doesn't contend with the live GPU training job."""
import json
from functools import partial
import torch
from torch.utils.data import DataLoader

import modeling.utils as U
U.DEVICE = torch.device('cpu')          # force CPU for this isolated test
from modeling.dataloader import BatchProcessor
from modeling.dataset import Dataset
from modeling.models import TigerModel, CorrectItemsLogitsProcessor
from modeling.utils import fix_random_seed

fix_random_seed(42)
cfg = json.load(open('./configs/tiger_ml1m_config.json'))
num_cb = cfg['dataset']['num_codebooks']; cb = cfg['model']['codebook_size']

ds = Dataset.create(inter_json_path=cfg['dataset']['inter_json_path'],
                    max_sequence_length=cfg['dataset']['max_sequence_length'],
                    sampler_type=cfg['dataset']['sampler_type'], is_extended=True)
train_sampler, _, _ = ds.get_samplers()
bp = BatchProcessor.create(cfg['dataset']['index_json_path'], num_cb, cfg['model']['user_ids_count'],
                           user_embeddings_path=cfg['dataset'].get('user_embeddings_path'))
dl = DataLoader(train_sampler, batch_size=32, drop_last=True, shuffle=True, collate_fn=bp)

model = TigerModel(embedding_dim=cfg['model']['embedding_dim'], codebook_size=cb, sem_id_len=num_cb,
    user_ids_count=cfg['model']['user_ids_count'], num_positions=cfg['model']['num_positions'],
    num_heads=cfg['model']['num_heads'], num_encoder_layers=cfg['model']['num_encoder_layers'],
    num_decoder_layers=cfg['model']['num_decoder_layers'], dim_feedforward=cfg['model']['dim_feedforward'],
    num_beams=cfg['model']['num_beams'], num_return_sequences=cfg['model']['top_k'],
    activation=cfg['model']['activation'], d_kv=cfg['model']['d_kv'], dropout=0.0,  # no dropout -> pure memorization
    layer_norm_eps=cfg['model']['layer_norm_eps'], initializer_range=cfg['model']['initializer_range'],
    logits_processor=partial(CorrectItemsLogitsProcessor, num_cb, cb, cfg['dataset']['index_json_path'],
                             cfg['model']['num_beams']),
    user_attr_dim=cfg['model'].get('user_attr_dim')).to(U.DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch = next(iter(dl))                 # ONE fixed batch, reused every step
assert 'user_attr.embedding' in batch, 'attribute path not active!'
print('overfitting a single fixed batch of 32 (attribute path). Expect loss -> ~0')
model.train()
for step in range(401):
    out = model(batch)
    loss = out['loss']
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        # teacher-forced next-token accuracy on the same batch
        with torch.no_grad():
            logits = out['logits']                      # (B, T, vocab)
            labels = batch['semantic_labels.ids'].reshape(logits.size(0), num_cb)
            offs = cb * torch.arange(num_cb)
            tgt = labels + offs                         # offset space
            pred = logits.argmax(-1)                    # (B, T)
            acc = (pred == tgt).float().mean().item()
        print(f'  step {step:3d} | loss {loss.item():.4f} | token-acc {acc:.3f}')
print('DONE - loss should be near 0 and token-acc near 1.0 if the pipeline is correct')
