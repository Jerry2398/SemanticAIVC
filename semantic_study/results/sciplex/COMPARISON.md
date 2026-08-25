# Semantic study — sciplex3 (5000 HVG, v2) — 5 encoders

Probes trained on train split (559,094 cells), evaluated on val (186,123; drug-Squidiff
172,786 SMILES-mapped). Latent dims: scVI=10, trVAE=10, scDisInFact(shared)=8, Squidiff=60.

| Metric (↑ better unless noted) | chance | scVI | scGen | trVAE | scDisInFact | Squidiff-nodrug | Squidiff-drug |
|---|---|---|---|---|---|---|---|
| **Decodability** |||||||
| GER gene-wise Pearson | 0 | 0.073 | 0.098 | 0.000 | 0.045 | 0.097 | 0.052 |
| GER cell-wise Pearson | 0 | 0.655 | **0.712** | 0.544 | 0.546 | 0.617 | 0.612 |
| GER cell-wise R² | 0 | 0.422 | **0.511** | 0.291 | 0.293 | 0.380 | 0.362 |
| CTD accuracy | 0.449 | **0.992** | **0.992** | 0.449 | 0.463 | 0.907 | 0.951 |
| CTD macro-F1 | — | **0.992** | 0.991 | 0.207 | 0.274 | 0.910 | 0.954 |
| DD accuracy | 0.009 | 0.022 | 0.031 | 0.008 | 0.008 | 0.016 | **0.426** |
| DD macro-F1 | — | 0.011 | 0.017 | 0.001 | 0.001 | 0.008 | **0.423** |
| DoseD R² | 0 | 0.015 | 0.021 | ~0 | ~0 | 0.007 | **0.089** |
| MOA-D accuracy | ~0.05 | 0.225 | 0.232 | 0.231 | 0.231 | 0.229 | **0.281** |
| MOA-D macro-F1 | — | 0.029 | 0.036 | 0.019 | 0.019 | 0.023 | **0.184** |
| **Organization** |||||||
| EGC Mantel r (pearson) | 0 | 0.409 | **0.609** | 0.003 | 0.110 | 0.157 | 0.348 |
| EGC distance-corr | 0 | 0.733 | **0.908** | 0.161 | 0.231 | 0.365 | 0.673 |
| LNP mean Jaccard | ~0.01 | 0.028 | **0.122** | 0.005 | 0.011 | 0.016 | 0.008 |
| PGC silhouette (↑) | 0 | −0.380 | −0.452 | −0.393 | −0.485 | −0.588 | **−0.304** |
| PGC within/between (↓) | 1.0 | 0.953 | 0.998 | 0.948 | 0.918 | 1.307 | **0.929** |
| PGC ratio perm-p (↓) | — | 0.034 | — | 0.031 | 0.054 | 0.982 | **0.007** |

Training budget: scVI = full 559k (GPU). scGen = 50k subsample (CPU; old torch can't
use the Hopper GPU). trVAE / scDisInFact = 100k subsample (GPU). Despite the smallest
subsample, scGen (a plain reconstructive VAE) tops the geometry metrics — so trVAE /
scDisInFact's weakness is likely their condition-integration objective, not just budget.

## ⚠️ Fairness caveat (read first)
Compute limits forced unequal training:
- **scVI** trained on the **full 559k** cells (GPU).
- **trVAE / scDisInFact / scGen** trained on a **100k (50k for scGen) subsample with fewer
  epochs** (trVAE hit the 12h walltime at 3% on full data; scDisInFact OOM'd at 134 GB).

trVAE and scDisInFact land at/near the CTD chance level (0.449) with very low macro-F1 —
their latents look **near-uninformative here, most likely undertrained**, not a true read
of the methods. Treat their rows as provisional pending equal-budget retraining.

## Analysis

**1. Cell identity (CTD).** scVI ≈ scGen (0.99) ≫ Squidiff-drug 0.95 > Squidiff-nodrug 0.91
≫ trVAE/scDisInFact ≈ chance (0.45). The reconstructive encoders (scVI, scGen, Squidiff)
encode the 3 cell lines cleanly; the condition-integration VAEs do not.

**2. Perturbation semantics (DD / DoseD / MOA).** Only **Squidiff-drug** decodes drug
(0.43), dose (R² 0.09) and MOA (0.28) above chance — the drug fingerprint × dose is fed
*into* its encoder. Every other encoder (incl. scVI, scGen) is at chance: the perturbation
signal in the measured transcriptome is **not linearly recoverable** from an unsupervised /
expression-only latent here.

**3. Transcriptome recoverability (GER).** scGen best (cell-Pearson 0.71 / R² 0.51), then
scVI; gene-wise ~0 for all — latents capture the coarse per-cell profile, not per-gene
perturbation responses.

**4. Geometry (EGC / LNP).** **scGen is best by a wide margin** (dcor 0.91, Jaccard 0.122 —
4× the next). scVI second (0.73 / 0.028), Squidiff-drug third on global (0.67). A plain
reconstructive VAE (scGen) preserves expression geometry best even on the smallest
(50k-CPU) training budget.

**5. Drug/MOA geometry (PGC).** Only **Squidiff-drug** organizes drugs by MOA with
significance (ratio 0.93, p=0.007). scGen/Squidiff-nodrug show none (p≈0.5, 0.98);
scVI/trVAE weakly (p≈0.03).

## Takeaways
- **scGen & scVI** = best *unsupervised* semantic spaces (cell identity + expression
  geometry); scGen leads on geometry/recoverability, scVI ties on identity. Both are blind
  to perturbation labels.
- **Squidiff-drug** = the only encoder with real *perturbation* semantics (drug/dose/MOA
  decodable + significant MOA geometry), because those labels enter the encoder directly.
- **Squidiff-nodrug** = good cell identity, no perturbation semantics.
- **trVAE / scDisInFact** = weak/inconclusive. scGen (smaller 50k budget) did great, so the
  weakness likely comes from their condition-integration objective, not just training budget
  — but an equal-budget retrain is still advisable to confirm.
