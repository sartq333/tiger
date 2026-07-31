# Experiments — ML-1M user-attribute conditioning & gender-bias probe

Scripts backing the ML-1M extension: TIGER conditioned on LLaMA-2-7B embeddings of
user demographics (`"Gender: X. Age: Y. Occupation: Z."`) projected into the T5
encoder as an extra token, instead of the original hashed user-ID.

**Run all scripts from the `tiger/tiger` directory** (same convention as the README),
with the package on the path:

```bash
cd tiger/tiger
export PYTHONPATH=$PWD
```

Prerequisites produced by `notebooks/usersPreprocessing.ipynb`:
`../data/ml-1m/{inter.json, index_rqvae.json, user_embeddings.pkl, item_id_map.json}`.
Training writes `../checkpoints/tiger_ml1m_best.pth`.

## Scripts

| Script | Purpose | Example |
|---|---|---|
| `smoke.py` | End-to-end sanity: build model, one train step + one beam-search validation batch; times the expensive path | `python experiments/smoke.py` |
| `overfit_test.py` | Overfit one fixed batch (dropout off) — loss must collapse to ~0 / token-acc→1.0, else the pipeline is broken. Runs on CPU. | `python experiments/overfit_test.py` |
| `eval_sample.py` | Full ranking metrics (NDCG/HR @5,10,20) on a random test-set sample | `python experiments/eval_sample.py --ckpt ../checkpoints/tiger_ml1m_best.pth --n 800 --device cuda` |
| `gender_gap.py` | **The gender probe.** Same watch history, condition as Male vs Female, compare top-K recommendations (+ optional aggregate genre-shift over N users) | `python experiments/gender_gap.py --ckpt ../checkpoints/tiger_ml1m_best.pth --device cuda --agg 100` |

`gender_gap.py` flags: `--ckpt` (checkpoint), `--device` (cuda/cpu), `--users`
(comma-sep user ids; auto-picks 3 with both genders present otherwise),
`--agg N` (aggregate genre-shift over N random users; 0 = skip).

## Key results (final checkpoint, NDCG@20 ≈ 0.167, 18 epochs)

Test metrics on an 800-user sample — for reference the repo's Amazon TIGER numbers
are ~3.5× lower (ML-1M is far denser):

| NDCG@5 | NDCG@10 | NDCG@20 | HR@5 | HR@10 | HR@20 |
|--:|--:|--:|--:|--:|--:|
| 0.102 | 0.121 | 0.141 | 0.141 | 0.203 | 0.281 |

Gender bias (aggregate over diverse users, Male→Female genre-share shift):
Romance **+1.2pp**, Comedy/Drama/Thriller up; Action **−0.8pp**, Adventure/Sci-Fi/
Western down — stereotype-consistent. Effect is largest on users with mixed
histories; for strongly genre-typed users it shows mostly as re-ranking within the
same top-K.

`logs/train_ml1m.log` is the training log for the run these numbers come from.
