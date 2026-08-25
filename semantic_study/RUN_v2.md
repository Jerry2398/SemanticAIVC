# Running the v2 multi-encoder study

Canonical dataset: `sciplex3_hvg5000_{train,val}.h5ad` (559094 / 186123 cells,
5000 HVG, `.X`=log-norm + `.layers['counts']`=raw). All encoders train on train,
embed val; metrics compare on val.

## Environments (one per model)

A single env for all four packages is not possible (scvi-tools/scgen/scarches
need Python >=3.10 and pin conflicting deps; scDisInFact is GitHub-only). So each
model has its own Python-3.10 + CUDA-torch env (built by `setup_envs.sh`), each
also carrying scanpy/sklearn so extraction AND metrics run in it:

```
/scratch/yuchen.yan/envs/sem_scvi          scvi-tools
/scratch/yuchen.yan/envs/sem_scgen         scgen
/scratch/yuchen.yan/envs/sem_trvae         scarches
/scratch/yuchen.yan/envs/sem_scdisinfact   scDisInFact (from GitHub)
```

`submit_encoder.sh` activates the right env automatically based on `MODEL`.

## Latent dims (official defaults)

scVI=10, scGen=100, trVAE=10, scDisInFact shared Ks=[8,...]. Squidiff: nodrug
(expression-only) / drug variants. Metrics are latent-dim agnostic; dims are
reported in each results json.

## One command per encoder (GPU PBS jobs, run in parallel)

```bash
cd semantic_study
qsub -v MODEL=scvi        submit_encoder.sh
qsub -v MODEL=scgen       submit_encoder.sh
qsub -v MODEL=trvae       submit_encoder.sh
qsub -v MODEL=scdisinfact submit_encoder.sh
```

Each job: trains the model (early-stopping on val loss), saves a checkpoint to
`/scratch/.../models/<model>/`, writes embeddings to the `embeddings/` dir, then
runs all 9 metrics -> `results/<model>/summary.json`.

## Squidiff (two variants, in the Squidiff repo)

```bash
cd ../Squidiff
qsub submit_train_v2_nodrug.sh          # expression-only  -> models/squidiff_nodrug
qsub submit_train_v2_drug.sh            # drug-structure   -> models/squidiff_drug
# after training:
cd ../semantic_study
python extract/extract_squidiff_embeddings.py --variant nodrug
python run_all.py --encoder squidiff_nodrug
python extract/extract_squidiff_embeddings.py --variant drug     # needs prep_squidiff_drug.py first
python run_all.py --encoder squidiff_drug
```

## Compare

Each encoder's numbers land in `results/<encoder>/summary.json`. Line them up to
compare Semantic Decodability (GER/CTD/DD/DoseD/MOA-D) and Organization
(EGC/LNP/PGC) across scVI / scGen / trVAE / scDisInFact / Squidiff(nodrug,drug).
