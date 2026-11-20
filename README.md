# Multimodal Patient Retrieval

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

## Ethical scope

All cohorts, embeddings, clinical variables, and subtypes are simulated. The
system is a research demonstration, not a patient-matching service.
