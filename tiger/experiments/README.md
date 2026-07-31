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
| `gender_gap.py` | **The gender probe.** Same watch history, condition as Male vs Female, compare top-K recommendations. Ships with baked-in strong examples; `--scan` re-finds them | `python experiments/gender_gap.py --device cuda` |
| `genre_bias.py` | Genre-level analysis over N users: genre-share Male vs Female, top-1 genre split, stratified by history type | `python experiments/genre_bias.py --device cuda --n 300` |
| `plot_genre.py` | Renders `genre_distribution.png` (training-data genre mix + gender tilt) | `python experiments/plot_genre.py` |

`gender_gap.py` flags: `--ckpt`, `--device` (cuda/cpu), `--users` (comma-sep ids),
`--scan N` (scan N users and show the most gender-divergent ones), `--n-strong K`,
`--agg N` (aggregate genre-shift over N users; 0 = skip). With no `--users`/`--scan`
it shows `DEFAULT_STRONG_USERS` — examples pre-found for `tiger_ml1m_best.pth`.

### Strong examples (validated)

Same watch history, only the gender attribute flipped — the *kinds* of movies change:

| User | Conditioned **Male** | Conditioned **Female** |
|---|---|---|
| 1080 (56+, exec) | Conan the Barbarian, Star Trek, Logan's Run, Indiana Jones | You've Got Mail, Truth About Cats & Dogs, Fabulous Baker Boys |
| 3713 (25-34, exec) | Fight Club, Unforgiven, Misery, Three Kings | Bambi, Fantasia, Casablanca, Bug's Life |
| 4509 (25-34, exec) | Robocop, Devil's Advocate, Alien Nation | Ghost, Fried Green Tomatoes, Dave |

Aggregate (300 users, top-10 genre share, Female − Male): Romance **+1.2**, Comedy
**+1.3**, Drama +0.5 up; Action **−1.4**, Thriller −0.8, Crime/Horror −0.5 down —
stereotype-consistent. See `genre_distribution.png` (Panel A confirms the catalog
is 18-genre diverse, so the tilt is a learned bias, not a data artifact).

**Caveat — batch dependence:** beam-search generation is sensitive to batch
composition (FP non-associativity across batch size, amplified by 100-beam
constrained decoding), so a user's recs differ single-user vs in a large batch.
The strong examples and `gender_gap.py`'s displayed rankings use **single-user**
generation so they reproduce exactly; the aggregate direction is robust to this
(FP noise can't create a consistent stereotype-aligned tilt), but per-user
aggregate assignments are batch-dependent.

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
