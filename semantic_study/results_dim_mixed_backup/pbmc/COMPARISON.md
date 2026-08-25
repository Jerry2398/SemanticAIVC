# Semantic study — PBMC (Kang 2018 IFN-β) — 5 encoders

Data: kang.h5ad, 2000 HVG, train 18,623 / val 6,050, 8 cell types, condition=ctrl/stim.
Probes: train on train split, eval on val. Latent dims: scVI=10, scGen=100, trVAE=10,
scDisInFact(shared)=8, Squidiff=60. squidiff_nodrug retrained (lr_anneal_steps=15000,
after the 100k-step run collapsed ~step 20k on this small dataset).

Applicable metrics: GER, CTD, DD(binary ctrl/stim), EGC, LNP, COG. DoseD/MOA-D/PGC
N/A (single perturbation, no dose/MOA).

| Metric | scVI | scGen | trVAE | scDisInFact | Squidiff-nodrug |
|---|---|---|---|---|---|
| GER cell-Pearson | 0.728 | **0.803** | 0.704 | 0.692 | 0.678 |
| GER cell-R² | 0.533 | **0.648** | 0.493 | 0.477 | 0.447 |
| CTD acc (chance 0.456) | 0.948 | **0.974** | 0.819 | 0.828 | 0.591 |
| CTD macro-F1 | 0.900 | **0.954** | 0.473 | 0.591 | 0.255 |
| DD acc (chance 0.507) | **0.982** | 0.977 | 0.940 | 0.516 | 0.577 |
| DD macro-F1 | **0.982** | 0.977 | 0.940 | 0.505 | 0.577 |
| EGC distance-corr | 0.852 | **0.956** | 0.687 | 0.885 | 0.553 |
| EGC Mantel-r | 0.571 | **0.690** | 0.377 | 0.498 | 0.336 |
| LNP mean Jaccard | 0.102 | **0.213** | 0.038 | 0.034 | 0.037 |
| **COG Mantel-r** (↑) | 0.525 | 0.463 | 0.463 | **0.534** | 0.450 |
| **COG Mantel-p** (↓) | **0.002** | 0.014 | 0.017 | 0.015 | 0.021 |

## Analysis

**COG (the new PBMC-only metric — latent vs Cell-Ontology tree).** ALL encoders show
**significant** COG (Mantel-r 0.45–0.53, p<0.025): every latent arranges the 8 cell-type
centroids roughly consistently with the Cell Ontology. scDisInFact (0.534) and scVI (0.525,
p=0.002) lead. COG uses coarse centroids, so even the weak Squidiff (0.450) still passes.

**Cell identity (CTD).** scGen (0.974 / F1 0.954) ≈ scVI (0.948 / 0.900) ≫ scDisInFact (0.828)
≈ trVAE (0.819) > Squidiff-nodrug (0.591 / F1 0.255). The dedicated scRNA VAEs encode the 8
PBMC types best; Squidiff's expression-only diffusion autoencoder is weakest.

**Perturbation (DD = ctrl vs stim).** IFN-β is a huge transcriptional shift → scVI/scGen/trVAE
decode it ~0.94–0.98. **scDisInFact ≈ chance (0.516) BY DESIGN**: its *shared* factor is meant
to be condition-invariant, so ctrl/stim is (correctly) removed from z_c. Squidiff-nodrug weak (0.58).

**Expression geometry (GER/EGC/LNP).** scGen best across all (GER-R² 0.65, EGC dcor 0.96, LNP
0.213 — ~2× the next). scVI second; scDisInFact high on EGC (0.885) but low on LNP (local).

## Takeaways
- **scGen** = best PBMC representation overall (identity + expression geometry + local structure).
- **scVI** = close second, best DD and most-significant COG.
- **scDisInFact** = shared factor correctly condition-invariant (DD≈chance) yet keeps cell-type
  geometry (high EGC/COG) — a clean illustration of its disentangling objective.
- **Squidiff-nodrug** = weakest on PBMC identity; expression-only diffusion AE < dedicated scRNA VAEs.
- **COG is now informative** (was N/A for sciplex): all latents are ontology-consistent, scVI/scDisInFact best.
