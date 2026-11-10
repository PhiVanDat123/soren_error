# Additional paper optimizers

Sources read for this implementation:

* [Spectrally Targeted Muon, arXiv:2610.10965v1](https://arxiv.org/abs/2610.10965),
  Algorithm 1 and Section 3.2.
* [Muon is Not That Special, arXiv:2605.11181v1](https://arxiv.org/abs/2605.11181),
  Algorithm 2, Sections 2.2 and 2.4, and Appendix B.

## Run Llama-60M

Run from `llama_pretraining` in the usual Linux/CUDA training environment.
Set the WandB credentials/name and GPU allocation in the launchers as usual.
These launchers use the existing 5000-step, seed-5, two-GPU configuration.

```bash
bash scripts/60M/TargetedMuonTop.sh
bash scripts/60M/TargetedMuonBottom.sh
bash scripts/60M/Kaon.sh
bash scripts/60M/Freon.sh
bash scripts/60M/Freon_3_4.sh
bash scripts/60M/TruncatedSGD.sh
```

All outputs go to separate `outputs/llama_60m_<variant>` directories.
The launchers do not enable approximation error logs or spectral NPZ logs.
Ordinary training/evaluation logging and model checkpoints still work.

| CLI optimizer | Transform | Options and defaults |
|---|---|---|
| `targeted_muon_top` | Orthogonalize modes above absolute threshold; retain raw momentum below | `--targeted_threshold 0.01 --spectral_steps 20` |
| `targeted_muon_bottom` | Retain raw momentum above threshold; orthogonalize below | Same |
| `kaon` | Five chaotic polynomial iterations, divide output by 1.175 | `--spectral_steps 5` |
| `freon` | SVD implementation of mean-Schatten-normalized spectral power | `--freon_exponent 0.6666666666666666 --freon_relative_floor 1e-6` |
| `truncated_sgd` | Remove top 5% of singular values, preserve original update Frobenius norm | `--truncate_fraction 0.05` |

`Freon_3_4.sh` uses exponent 0.75. `--lrmuon` controls hidden matrix updates;
`--lr` controls auxiliary AdamW. Initial learning rates are inherited from the
Muon launcher as starting points, not tuned results from these papers.
Tune LR and the absolute targeted threshold on the Llama setup before drawing
benchmark conclusions. If almost every singular value is on one side of the
threshold, the targeted update approaches raw momentum or full polar update.

## Algorithm and integration details

`paper_optimizers.py` implements a common optimizer using the repository's
`use_muon` group flag. For these new methods, hidden transformer matrices use
the selected spectral method; embeddings, LM head and gain/bias vectors use
AdamW. This differs from the current existing Muon/Soren grouping, which also
assigns embeddings and head to the spectral group. Account for that distinction
in controlled comparisons. Tied embedding/head parameters are deduplicated.

Targeted Muon uses **unnormalized** Nesterov momentum, as in Algorithm 1:
`B <- beta*B + grad; M <- grad + beta*B`. The threshold is absolute on `M`,
before any Frobenius normalization in the sign iteration. The same momentum
convention is used for all new methods. Defaults: beta=0.95 for spectral
methods, beta=0.9 for TruncatedSGD; override with `--paper_momentum`.

Targeted Muon applies the convergent polynomial
`p(x)=3x-(16/5)x^3+(6/5)x^5` for 20 fp32 steps to the shifted Gram matrix and
momentum. Smaller left-Gram projectors replace right-Gram projectors for wide
matrices, an algebraically equivalent implementation. Resolution close to the
threshold still depends on conditioning and the finite iteration count.

Kaon implements `X <- 4.1*(I-X@X.T)^2@X`, starting with Frobenius-normalized
momentum and returning `X/1.175`. It is deterministic for fixed inputs; it does
not sample fresh random singular values. Matrix operations use fp32.

Freon uses `q=2-2*c`, with singular coefficients
`(sigma / power_mean_q(sigma))^(q-1)`. At q=0 the power mean is the geometric
mean. This gives normalized SGD at c=0, polar at c=0.5 and an inverted spectrum
at c>0.5. Small singular values are floored relative to sigma_max, and exact
zero modes remain zero (regularized extension at rank deficiency).
**This implementation uses SVD, not the paper's rational/QDWH acceleration.**
Do not report its runtime as a reproduction of the paper's accelerated Freon.
`--spectral_steps` has no effect on this SVD backend.

TruncatedSGD removes `ceil(fraction * min(shape))` leading modes and rescales
the remaining update to the original Frobenius norm. If nothing remains, the
update is zero. Kaon and Freon retain the repository's aspect-ratio scale
`sqrt(max(1, rows/cols))`; targeted Muon and TruncatedSGD retain their raw-update
scales. This convention should be held fixed or explicitly noted in comparisons.

For distributed pretraining, expensive matrix transforms are assigned by
parameter index to ranks and updated parameters are broadcast. Momentum states
are maintained on every rank so the training entry point's rank-0 checkpoint
contains all states. Optimizer hyperparameters are in parameter groups and
survive state_dict loading. No approximation logger is attached.

## Validation

```bash
python -m unittest test_paper_optimizers -v
```

Tests cover threshold selection, transpose handling, scalar Kaon equivalence,
Freon spectral powers, truncation/norm preservation, zero matrices, auxiliary
AdamW and checkpoint continuation. CUDA/DDP training and benchmark tuning have
not been run in this CPU environment.
