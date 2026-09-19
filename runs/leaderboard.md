# Leaderboard

Generated 2026-09-19T01:39+00:00 from runs/*/result.json. Selection on development partitions only; the locked test is never read here.

- **Qwen/Qwen3-0.6B-Base** incumbent: transfer 0.592, dev 0.812, seeds [0, 0] (auto-06b-r1/00-trial-0, v4-06b-hardened/00-trial-0)
- **Qwen/Qwen3-4B-Base** incumbent: transfer 0.735, dev 0.834, seeds [1] (v4-4b-baseline/01-trial-1)

| study/trial | base | seed | dev acc | transfer acc | Brier | conf-err | held-out pairs | none_present | perm flip | gates | $ | knobs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v3-8b-s0/00-trial-0 | Qwen3-8B-Base | 0 | 0.843 | 0.765 | 0.377 | 0.104 | 0.56 | 0.83 | 0.02 | fail | 1.45 | accum=4, epochs=2, train_sources=banking77,boolq,agnews,mnli,sst5,yelp,trec,dbpedia14,amazon,imdb,legacy_policy |
| v3-data-capacity-s1/03-trial-3 | Qwen3-4B-Base | 1 | 0.840 | 0.750 | 0.357 | 0.052 | 0.62 | 0.82 | 0.00 | fail | 1.09 | epochs=2, train_sources=banking77,boolq,agnews,mnli,sst5,yelp,trec,dbpedia14,amazon,imdb,compositional |
| v3-data-capacity-s0/03-trial-3 | Qwen3-4B-Base | 0 | 0.825 | 0.747 | 0.382 | 0.088 | 0.52 | 0.83 | 0.08 | fail | 0.89 | epochs=2, train_sources=banking77,boolq,agnews,mnli,sst5,yelp,trec,dbpedia14,amazon,imdb,compositional |
| v3-8b-s0/01-trial-1 | Qwen3-8B-Base | 0 | 0.840 | 0.741 | 0.375 | 0.082 | 0.61 | 0.83 | 0.02 | fail | 1.63 | accum=4, epochs=2, train_sources=banking77,boolq,agnews,mnli,sst5,yelp,trec,dbpedia14,amazon,imdb,compositional |
| v4-4b-baseline/01-trial-1 | Qwen3-4B-Base | 1 | 0.834 | 0.735 | 0.373 | 0.052 | 0.47 | 0.85 | 0.02 | fail | 2.84 | epochs=2, p_none_pair=0.25 |
| v3-data-capacity-s0/02-trial-2 | Qwen3-4B-Base | 0 | 0.824 | 0.721 | 0.409 | 0.088 | 0.45 | 0.82 | 0.03 | fail | 0.89 | epochs=2, train_sources=banking77,boolq,agnews,mnli,sst5,yelp,trec,dbpedia14,amazon,imdb,legacy_policy |
| v3-data-capacity-s1/02-trial-2 | Qwen3-4B-Base | 1 | 0.826 | 0.721 | 0.414 | 0.116 | 0.47 | 0.80 | 0.03 | fail | 0.94 | epochs=2, train_sources=banking77,boolq,agnews,mnli,sst5,yelp,trec,dbpedia14,amazon,imdb,legacy_policy |
| v4-4b-baseline/00-trial-0 | Qwen3-4B-Base | 0 | 0.849 | 0.704 | 0.463 | 0.113 | 0.58 | 0.93 | 0.02 | fail | 3.02 | epochs=2, p_none_pair=0.25 |
| ablation-v2/07-trial-7 | Qwen3-0.6B-Base | 1 | 0.793 | 0.621 | 0.532 |  | 0.07 | 0.74 | 0.04 | pass | 0.31 | epochs=2 |
| ablation-v2/06-trial-6 | Qwen3-0.6B-Base | 0 | 0.816 | 0.620 | 0.525 |  | 0.03 | 0.76 | 0.06 | pass | 0.34 | epochs=2 |
| arch-06b/06-trial-6 | Qwen3-0.6B-Base | 0 | 0.799 | 0.610 | 0.519 | 0.062 | 0.14 | 0.85 | 0.02 | fail | 0.31 | epochs=2, lr=0.0001, p_none_pair=0.25 |
| ablation-v2/04-trial-4 | Qwen2.5-0.5B | 0 | 0.742 | 0.605 | 0.501 |  | 0.05 | 0.49 | 0.12 | fail | 0.27 | epochs=2 |
| v4-06b-hardened/02-trial-2 | Qwen3-0.6B-Base | 2 | 0.800 | 0.605 | 0.545 | 0.066 | 0.06 | 0.85 | 0.02 | fail | 0.8 | epochs=2, p_none_pair=0.25 |
| arch-06b/03-trial-3 | Qwen3-0.6B-Base | 0 | 0.799 | 0.599 | 0.538 | 0.050 | 0.19 | 0.82 | 0.00 | fail | 0.54 | epochs=2, option_isolation=1, p_none_pair=0.25, special_embeddings=1 |
| v3-data-capacity-s1/00-trial-0 | Qwen3-0.6B-Base | 1 | 0.760 | 0.599 | 0.549 | 0.082 | 0.06 | 0.68 | 0.10 | fail | 0.36 | epochs=2, train_sources=banking77,boolq,agnews,mnli,sst5,yelp,trec,dbpedia14,amazon,imdb,legacy_policy |
| ablation-v2/01-kev2 | legacy |  | 0.725 | 0.598 | 0.525 |  | 0.00 | 0.69 | 0.06 | pass | 0.07 | defaults |
| v4-06b-hardened/00-trial-0 | Qwen3-0.6B-Base | 0 | 0.805 | 0.598 | 0.521 | 0.052 | 0.11 | 0.78 | 0.02 | fail | 0.8 | epochs=2, p_none_pair=0.25 |
| auto-06b-r1/02-trial-2 | Qwen3-0.6B-Base | 0 | 0.807 | 0.596 | 0.542 | 0.067 | 0.03 | 0.83 | 0.00 | fail | 1.15 | epochs=2, p_none_pair=0.25, perm_kl=0.5 |
| v3-data-capacity-s1/01-trial-1 | Qwen3-0.6B-Base | 1 | 0.750 | 0.596 | 0.532 | 0.044 | 0.03 | 0.67 | 0.05 | fail | 0.36 | epochs=2, train_sources=banking77,boolq,agnews,mnli,sst5,yelp,trec,dbpedia14,amazon,imdb,compositional |
| arch-06b/05-trial-5 | Qwen3-0.6B-Base | 0 | 0.794 | 0.595 | 0.500 | 0.030 | 0.19 | 0.83 | 0.00 | fail | 0.31 | epochs=2, lr=0.0003, p_none_pair=0.25 |
| v4-06b-hardened/01-trial-1 | Qwen3-0.6B-Base | 1 | 0.793 | 0.595 | 0.579 | 0.082 | 0.09 | 0.80 | 0.03 | fail | 0.83 | epochs=2, p_none_pair=0.25 |
| v3-data-capacity-s0/01-trial-1 | Qwen3-0.6B-Base | 0 | 0.757 | 0.593 | 0.531 | 0.064 | 0.06 | 0.67 | 0.03 | fail | 0.31 | epochs=2, train_sources=banking77,boolq,agnews,mnli,sst5,yelp,trec,dbpedia14,amazon,imdb,compositional |
| arch-06b/01-trial-1 | Qwen3-0.6B-Base | 0 | 0.812 | 0.590 | 0.537 | 0.043 | 0.09 | 0.82 | 0.02 | fail | 0.53 | epochs=2, p_none_pair=0.25, special_embeddings=1 |
| arch-06b/02-trial-2 | Qwen3-0.6B-Base | 0 | 0.805 | 0.590 | 0.559 | 0.066 | 0.14 | 0.80 | 0.02 | fail | 0.32 | epochs=2, head_dim=1024, p_none_pair=0.25 |
| auto-06b-r1/01-trial-1 | Qwen3-0.6B-Base | 0 | 0.808 | 0.590 | 0.542 | 0.053 | 0.14 | 0.85 | 0.03 | fail | 0.81 | epochs=2, p_none_pair=0.25, special_embeddings=0 |
| auto-06b-r1/07-trial-7 | Qwen3-0.6B-Base | 0 | 0.806 | 0.588 | 0.563 | 0.081 | 0.06 | 0.87 | 0.00 | fail | 0.78 | epochs=2, p_none_distract=0.25 |
| auto-06b-r1/00-trial-0 | Qwen3-0.6B-Base | 0 | 0.820 | 0.587 | 0.560 | 0.078 | 0.06 | 0.83 | 0.00 | fail | 0.8 | epochs=2, p_none_pair=0.25 |
| arch-06b/00-trial-0 | Qwen3-0.6B-Base | 0 | 0.800 | 0.581 | 0.587 | 0.098 | 0.06 | 0.82 | 0.00 | fail | 0.33 | epochs=2, option_isolation=1, p_none_pair=0.25 |
| ablation-v2/00-kev-0.5b | legacy |  | 0.712 | 0.575 | 0.514 |  | 0.20 | 0.62 | 0.06 | pass | 0.07 | defaults |
| auto-06b-r1/06-trial-6 | Qwen3-0.6B-Base | 0 | 0.816 | 0.575 | 0.514 | 0.047 | 0.05 | 0.80 | 0.02 | fail | 1.13 | epochs=2, option_isolation=0, p_none_pair=0.25, perm_kl=0.2 |
| auto-06b-r1/05-trial-5 | Qwen3-0.6B-Base | 0 | 0.771 | 0.572 | 0.535 | 0.034 | 0.02 | 0.72 | 0.08 | fail | 0.51 | p_none_pair=0.25 |
| auto-06b-r1/03-trial-3 | Qwen3-0.6B-Base | 0 | 0.768 | 0.569 | 0.515 | 0.020 | 0.14 | 0.77 | 0.03 | fail | 0.93 | epochs=2, lora=64, ord_w=0.5, p_none_pair=0.25 |
| ablation-v2/03-trial-3 | Qwen2.5-0.5B | 0 | 0.722 | 0.562 | 0.590 |  | 0.00 | 0.64 | 0.10 | fail | 0.2 | epochs=2, train_sources=banking77,boolq,agnews,mnli,sst5,yelp,trec,dbpedia14,amazon,imdb |
| auto-06b-r1/04-trial-4 | Qwen3-0.6B-Base | 0 | 0.763 | 0.553 | 0.516 | 0.003 | 0.08 | 0.80 | 0.03 | fail | 0.9 | epochs=2, lr=0.0005, ord_w=0.25, p_none_pair=0.25 |
| v3-data-capacity-s0/00-trial-0 | Qwen3-0.6B-Base | 0 | 0.722 | 0.546 | 0.584 | 0.023 | 0.03 | 0.75 | 0.05 | fail | 0.29 | epochs=2, train_sources=banking77,boolq,agnews,mnli,sst5,yelp,trec,dbpedia14,amazon,imdb,legacy_policy |
| ablation-v2/05-trial-5 | Qwen2.5-0.5B | 1 | 0.657 | 0.482 | 0.580 |  | 0.05 | 0.44 | 0.14 | fail | 0.22 | epochs=2 |
| ablation-v2/02-trial-2 | Qwen2.5-0.5B | 0 | 0.552 | 0.466 | 0.614 |  | 0.00 | 0.22 | 0.25 | pass | 0.15 | epochs=2, train_sources=banking77,boolq,agnews,mnli,sst5,yelp |
| v3-smoke-02/01-trial-1 | Qwen3-4B-Base | 0 | 0.421 | 0.349 | 1.140 | 0.363 | 0.00 | 0.10 | 0.70 | fail | 0.1 | train_sources=banking77,boolq,agnews,mnli,sst5,yelp,trec,dbpedia14,amazon,imdb,legacy_policy |
| v3-smoke-02/00-trial-0 | Qwen3-0.6B-Base | 0 | 0.474 | 0.341 | 0.918 | 0.174 | 0.00 | 0.10 | 0.90 | fail | 0.05 | train_sources=banking77,boolq,agnews,mnli,sst5,yelp,trec,dbpedia14,amazon,imdb,compositional |
| backbone-v1/00-kev-0.5b | legacy |  | 0.797 |  |  |  |  | 0.78 | 0.03 | pass | 0.03 | defaults |
| backbone-v1/01-trial-1 | Qwen2.5-0.5B | 0 | 0.688 |  |  |  |  | 0.42 | 0.14 | pass | 0.09 | epochs=2 |
| backbone-v1/02-trial-2 | Qwen3-0.6B-Base | 0 | 0.725 |  |  |  |  | 0.42 | 0.06 | pass | 0.15 | epochs=2 |
| backbone-v1/03-trial-3 | Qwen2.5-0.5B | 1 | 0.697 |  |  |  |  | 0.36 | 0.11 | fail | 0.11 | epochs=2 |
| backbone-v1/04-trial-4 | Qwen3-0.6B-Base | 1 | 0.721 |  |  |  |  | 0.22 | 0.17 | fail | 0.15 | epochs=2 |
| gatecheck-trial4/00-checkpoint | legacy |  | 0.721 |  |  |  |  | 0.22 | 0.17 | pass | 0.03 | defaults |
| research-smoke-runner-v2/00-trial-0 | Qwen2.5-0.5B | 0 | 0.389 |  |  |  |  | 0.33 | 1.00 | pass |  | ord_w=0.1, perm_frac=1, perm_kl=0.1 |
| research-smoke-runner-verified/00-trial-0 | Qwen2.5-0.5B | 0 | 0.389 |  |  |  |  | 0.33 | 1.00 | pass |  | accum=3, ord_w=0.1, perm_frac=1, perm_kl=0.1 |
| smoke/00-trial-0 | Qwen2.5-0.5B | 0 | 0.389 |  |  |  |  | 0.33 | 1.00 | pass | 0.04 | ord_w=0.1, perm_frac=1, perm_kl=0.1 |
| smoke-arch/00-trial-0 | Qwen2.5-0.5B | 0 | 0.222 |  |  |  |  | 0.17 | 0.00 | pass |  | head_dim=512, option_isolation=1, p_none_pair=0.5, special_embeddings=1 |
| smoke-mix/00-trial-0 | Qwen2.5-0.5B | 0 | 0.389 |  |  |  |  | 0.33 | 0.83 | pass |  | public_frac=0.5, synthetic_repeat=2 |
