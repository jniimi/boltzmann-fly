# Boltzmann-fly

**An energy-based world model with counterfactual inference, on the MaleCNS mushroom-body wiring** — a hobby project.

## What this is

[Purchase World](https://arxiv.org/abs/2605.07199) (Niimi, ICONIP 2026, arXiv:2605.07199) trains a Deep Boltzmann Machine (DBM) as an energy-based world model of simulated consumer behaviour and uses one frozen belief representation for three tasks: free-energy consistency scoring, outcome prediction, and counterfactual (do-intervention) uplift / CATE recovery.

This repository re-runs that experiment with the **coupling pattern of the DBM fixed to the wiring of the mushroom body in MaleCNS v1.0**, the complete connectome of the male *Drosophila* central nervous system released by HHMI Janelia's FlyEM team, the Cambridge Connectomics Group, and Google Research (v1.0: 2026-06-08; Berg et al., *Cell*, 2026; ~166,700 neurons). Only the existence of connections is taken from the connectome (a 0/1 mask); connection **magnitudes are learned** by persistent contrastive divergence. The directed graph is symmetrised so the model has an energy function. Data, tasks, hyperparameters and evaluation code are the original Purchase-World ones (vendored, MIT), so the original dense DBM can be reproduced as an anchor in the same pipeline.

```
@inproceedings{niimi2026purchaseworld,
    title = "Three-in-One World Model: Energy-Based Consistency, Prediction, and Counterfactual Inference for Marketing Intervention",
    author = "Niimi, Junichiro",
    booktitle = "Proceedings of the 33rd International Conference on Neural Information Processing (ICONIP 2026)",
    year = "2026",
    note = "arXiv: 2605.07199"
}
```

| Mushroom body | Role in the fly | This model |
|---|---|---|
| PN (projection neurons, 343 in the right hemisphere) | carry odour input into the mushroom body | visible layer; the 72 Purchase-World features are assigned to PN units |
| KC (Kenyon cells, ~2,000 per hemisphere; each receives ~5 PN inputs) | sparse high-dimensional expansion of the input | hidden layer 1 (1,918 KCs with at least one PN input) |
| MBON (mushroom body output neurons, 49) | read out the KC population toward approach/avoid behaviour | hidden layer 2 |

The goal was **"it runs"**: a faithful port that executes all three tasks on the real wiring, with controls so the result can be interpreted. It was not a goal to match or beat the original model, and nothing here is tuned toward that. Everything is one seed (data, mask assignment, training).

## Results (seed 0, test split)

**With all 72 features connected, the fly-masked Boltzmann machine runs and matches the original on prediction.** Visit AUC 0.712 vs 0.711 for the original dense DBM (raw-feature MLP 0.713); purchase AUC 0.678 vs 0.678 (MLP 0.679). Counterfactual recovery of the promotion-responsiveness parameter γ is at the original's level (Spearman +0.548 vs +0.564); recovery of the price-sensitivity parameter α is below it (+0.226 vs +0.639; degree-preserving control +0.419). The paper's free-energy clamp test does **not** reproduce its direction for the fly wiring (the clamped counterfactual receives a *lower* free energy in 98 % of rows) but does for a degree-preserving random rewiring of the same graph. No fly variant is claimed to beat the original; the one number where a control sits above it (degree-preserving γ +0.578 vs +0.564) is inside the ~0.06 seed noise of that metric.

| model | layers | couplings | visit AUC | purchase AUC | ρ γ (push→visit) | ρ α (sale1→purchase) | clamp ΔF mean | Welch t | train |
|---|---|---|---|---|---|---|---|---|---|
| V0: original dense DBM, reproduced here | 72-64-32-16 | 7,168 | 0.711 | 0.678 | +0.564 | +0.639 | +1.68 | −40.7 | 2.2 h CPU |
| ICONIP paper values (same data seed, different init) | 72-64-32-16 | 7,168 | 0.713 | 0.677 | +0.625 | +0.548 | +1.07 | −7.7 | 5.9 h MPS |
| **V1: fly wiring, 72/72 features connected** | 343-1918-49 | 35,165 | 0.712 | 0.678 | +0.548 | +0.226 | −12.63 | −11.6 | 9.3 h MPS |
| V1, degree-preserving random rewiring | 343-1918-49 | 35,165 | 0.712 | 0.679 | +0.578 | +0.419 | +9.06 | −38.6 | 9.3 h MPS |
| V1, naive assignment (only 20/72 features connected) | 343-1892-49 | 22,330 | 0.655 | 0.597 | −0.043 | +0.420 | +5.47 | +1.0 | 7.6 h |
| V1 naive, degree-preserving rewiring | 343-1892-49 | 22,330 | 0.656 | 0.597 | +0.048 | +0.349 | +3.35 | −1.7 | 8.9 h |
| V1 naive, Erdős–Rényi rewiring (connects 72/72) | 343-1892-49 | 22,330 | 0.712 | 0.679 | +0.373 | +0.365 | +0.72 | −33.3 | 11.4 h |
| V1b: 72 single PN neurons, naive (30/72) | 343-1892-49 | 22,330 | 0.677 | 0.629 | +0.171 | −0.273 | +2.53 | −22.9 | 10.2 h |
| V2: PN types, naive (20/72) | 72-1892-49 | 15,667 | 0.653 | 0.593 | +0.093 | +0.331 | +2.38 | +3.7 | 8.0 h |
| V2 naive, degree-preserving rewiring | 72-1892-49 | 15,667 | 0.654 | 0.590 | −0.063 | +0.252 | +1.85 | +2.6 | 8.3 h |

ρ = Spearman correlation between the adapter's logit-scale uplift and the true latent parameter of the simulation. Clamp = the paper's "purchasing without recent promotion" free-energy test; Welch t compares the penalty between high- and low-β consumers (negative = the paper's direction). Fly-variant times are for two or three runs sharing one Apple M2 Ultra GPU. Full per-run numbers: [`docs/results/summary.md`](docs/results/summary.md); the run JSONs are in `docs/results/runs/`.

### What the numbers mean

1. **Input coverage decides everything (a mask-construction bug in the first runs, fixed).** At synapse threshold 5 only 136 of the 343 PNs (68 of 177 PN types) contact any Kenyon cell. The first assignment drew feature→type at random from *all* types, so only 20 of the 72 features ever reached a KC ("naive" rows). Degree-preserving rewiring keeps those zero out-degrees, which is why it matched the fly wiring exactly, while Erdős–Rényi rewiring gives every PN targets and recovered the raw-feature baseline. The headline rows assign features only to KC-projecting types (`--type-select random-projecting --min-weight 1`; 87 projecting types at threshold 1), which connects 72/72.
2. **Sparse fan-in and maximum likelihood.** A KC sees 2–5 PN inputs, and a Boltzmann-machine coupling only grows when it captures dependence among a hidden unit's inputs. Features landing on one KC are essentially independent (median pairwise mutual information 0.000 nats), so most KCs stay near their bias-determined activity (median per-unit sd 0.01–0.02 vs 0.17 in V0) and the learned coupling strength of a KC tracks the mutual information among its inputs (Spearman +0.44 to +0.67 in every checkpoint). Fan-in-scaled initialisation and 10–100× larger learning rates do not change this ([`docs/step1_design.md`](docs/step1_design.md), §8). This is a non-degeneracy check, not tuning.
3. **Why the clamp test flips sign for the fly wiring.** Decomposing the free-energy change by layer, the PN–KC and KC-bias terms are nearly identical in the fly model and its degree-preserving control; the sign is decided by the KC→MBON block. In the control the MBON layer responds to the clamped KCs and penalises the counterfactual (+10.4); in the fly wiring the MBON layer is inert (half the MBONs saturated or constant) and contributes −3.1. MBON in-degrees are identical in the two masks, so the difference is the compartment-structured pattern of which KCs converge on which MBON (§10).
4. **Why α lags.** The α information is fully present in the fly belief (ridge R² of true α from the KC layer 0.74, equal to the original and to the raw features). The gap is in the purchase adapter's sale1 contrast, which is about three times smaller on the 1,967-dimensional belief than on the original's 112-dimensional one; fly wiring vs control cannot be separated from noise with one seed (§10).

Dale's-law (sign-fixed) couplings are implemented (`--dale`) and unit-tested but were not run. The single inhibitory APL neuron is stored in the masks but unused.

## Reproduce

```
uv sync --group dev
uv run python -m pytest -q                                   # 25 tests: masking, numerics, data identity
uv run python scripts/step0_extract_mb.py                    # MaleCNS -> mushroom-body subgraph + docs/step0_mb_stats.md
uv run python scripts/build_masks.py --variant all --control all --seed 0
uv run python scripts/build_masks.py --variant all --control all --seed 0 --min-weight 1 --type-select random-projecting
uv run python scripts/run_experiment.py --variant V0 --seed 0 --device cpu --threads 8                                          # ~2.5 h
uv run python scripts/run_experiment.py --variant V1 --seed 0 --device mps --min-weight 1 --type-select random-projecting       # headline, ~8 h
uv run python scripts/run_experiment.py --variant V1 --control degree --seed 0 --device mps --min-weight 1 --type-select random-projecting
uv run python scripts/summarize.py                           # docs/results/summary.md
```

- **Connectome.** MaleCNS v1.0 flat connectome, read from the public bucket `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/` (CC-BY 4.0; ~570 MB for the three files used). `step0_extract_mb.py` selects Traced neurons of class ALPN / Kenyon_Cell / MBON (plus APL) in the right hemisphere. Statistics of the subgraph: [`docs/step0_mb_stats.md`](docs/step0_mb_stats.md).
- **Data.** The Purchase-World panel (1,024 consumers × 365 days, lag window 4, seed 42) is regenerated by the vendored generator and is bit-identical to the ICONIP run (tested against the original val/test CSVs). Cached under `/Volumes/EXTERNAL/malecns/derived/boltzmann-fly/` (set `BOLTZMANN_FLY_DERIVED` to relocate).
- **Hyperparameters** are the ICONIP ones throughout: pretraining 100 epochs per RBM (Adam 1e-4, weight decay 1e-3, PCD k=1, batch 128), joint fine-tuning 300 epochs (Adam 1e-5, weight decay 1e-4, k=5, 10 mean-field iterations, patience 20 on validation reconstruction BCE), adapters and baseline MLP 64-32-16 (Adam 5e-4, dropout 0.1, batch 256, 100 epochs, patience 30).
- **CPU bit-identity (V0).** Two independent runs of `--variant V0 --seed 0 --device cpu --threads 8` (torch 2.14.0, `torch.use_deterministic_algorithms(True)`) produced identical metrics and byte-identical state dicts for every saved module. V0 runs 2.5× faster on CPU than on MPS; the fly variants run 4.5× faster on MPS, where determinism is not guaranteed. `--device cuda` is supported but untested.
- **Masking guarantee.** Masked couplings are exactly zero at initialisation and after every pretraining and fine-tuning step; an all-ones mask reproduces the upstream DBM numerically (unit tests).

## Weights

Trained checkpoints (largest 12 MB), the mask files they used, the run JSONs and logs are published as a release bundle with a `SHA256SUMS` manifest; see the Releases page of this repository. Checkpoints are pickled `torch.nn.Module` objects as written by the vendored training code and load with `pickle` after `import boltzmann_fly.masked_dbm`. The MaleCNS-derived edge tables are not redistributed; the mask files contain only 0/1 presence matrices and body ids.

## Layout

```
scripts/step0_extract_mb.py   MaleCNS flat connectome -> mushroom-body subgraph + statistics
scripts/build_masks.py        V1 / V1b / V2 masks, degree-preserving and Erdős–Rényi controls
scripts/run_experiment.py     one run: data, DBM training, belief, three tasks -> docs/results/runs/<run_id>.json
scripts/summarize.py          docs/results/summary.md
src/boltzmann_fly/             masks, masked DBM, pipeline; src/boltzmann_fly/vendor/ = Purchase World code (MIT, import lines only changed)
docs/step0_mb_stats.md        connectome subgraph statistics
docs/step1_design.md          design, exact ICONIP configuration, timing, results, diagnoses (§8–§10)
docs/results/                 results.csv, summary.md, per-run JSON
tests/                        25 unit tests
```

## Acknowledgements and citations

- MaleCNS v1.0: Berg, S. et al. *Sexual dimorphism in the complete Drosophila male central nervous system connectome.* Cell 189, 5504–5526 (2026). Data: Janelia FlyEM / Google Research / Cambridge Drosophila Connectomics Group, CC-BY 4.0. Project page: https://male-cns.janelia.org/ (contains neuron renderings).
- Purchase World: J. Niimi, *Three-in-One World Model: Energy-Based Consistency, Prediction, and Counterfactual Inference for Marketing Intervention*, ICONIP 2026 (arXiv:2605.07199).
- Design principle: J. Niimi, *the Mouth is Not the Brain*, ICLR 2026 Workshop on World Models (arXiv:2601.17094).
- Related: Lappalainen et al., *Connectome-constrained networks predict neural activity across the fly visual system*, Nature 2024; Dasgupta, Stevens & Navlakha, *A neural algorithm for a fundamental computing problem*, Science 2017 (the PN→KC expansion as a hashing architecture); Costi et al., *The Drosophila connectome as a computational reservoir for time-series prediction*, Biomimetics 2025.

## License

MIT for the code in this repository (see `LICENSE`). The vendored Purchase World code is MIT (same author). The MaleCNS data keeps its own CC-BY 4.0 license.
