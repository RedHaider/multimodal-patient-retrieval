# Multimodal Patient Retrieval: Missing-Modality Benchmark

An original, synthetic benchmark exploring AI-enabled precision medicine. It
retrieves similar patients by fusing an
imaging embedding with clinical variables and measures whether the retrieved
cohort shares the query patient's simulated disease subtype.

This project does not reproduce a published model and contains no patient data.

## Run

```powershell
python -m pip install -e .
python -m patient_retrieval.experiment --output outputs/demo
pytest -q
```

## Experiment

The benchmark compares image-only, clinical-only, and fused retrieval using
precision@5 and mean reciprocal rank. Fusion weights are tuned only on a
validation split, keeping the test set untouched.

## Heavy benchmark

The expanded experiment uses five independent seeds and separate 600-patient
gallery, 300-patient validation, and 400-patient test partitions. It compares a
validation-tuned global fusion baseline with an availability-aware fusion method
under clean, image-missing, clinical-missing, and mixed-missing conditions.
Fusion weights are selected on validation data only. The test set remains held
out, and results include precision@5, mean reciprocal rank, and 95% confidence
intervals.

```powershell
python -m patient_retrieval.heavy_experiment --output outputs/heavy
```

Any improvement claim is limited to this synthetic retrieval protocol. This
does not reproduce or improve a professor's published model and is not a
clinical patient-matching system.

### Verified result

Across five seeds, availability-aware fusion improved mixed-missing
precision@5 from **0.5838** to **0.5955** (+0.0117) and mean reciprocal rank
from **0.7352** to **0.7475** (+0.0123). It improved precision@5 under both
single-modality missing conditions as well. Clean precision@5 changed by only
+0.0008, so the evidence supports missing-modality robustness rather than a
general clean-data accuracy claim. Confidence intervals are reported in
`results/heavy_metrics.csv`.

## Ethical scope

All cohorts, embeddings, clinical variables, and subtypes are simulated. The
system is a research demonstration, not a patient-matching service.
