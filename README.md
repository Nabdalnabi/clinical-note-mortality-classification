# Clinical-Note Mortality Classification

Research code adapted from Nader Abdalnabi's breast-cancer clinical-note notebooks.
The implemented workflow uses a frozen Hugging Face BioDistilBERT encoder,
overlapping token windows, patient-level embedding aggregation, and a tuned
PCA/logistic-regression classifier. This is feature extraction, not language-model
fine-tuning or generative prognosis.

## Scientific scope

The notebook target is **death recorded versus no death recorded**. Missing death
documentation does not establish survival. This is not a time-to-event model,
does not address censoring, and does not estimate five-year mortality. No clinical
performance claims or original patient results are published in this sample.

The longer source notebook also explores `emilyalsentzer/Bio_ClinicalBERT` and
`yikuan8/Clinical-Longformer`. The smaller, focused notebook was selected as the
basis here; only `nlpie/bio-distilbert-uncased` is implemented in this release.
No claim is made that it outperforms the other encoders.

## Workflow

1. Validate one nonempty note per unique patient and explicit binary labels.
2. Encode locally with frozen BioDistilBERT using 512-token windows and 128-token overlap.
3. Mean-pool nonpadding tokens; aggregate chunks weighted by token counts.
4. Use five outer stratified folds and four inner folds. Scaling, PCA, and
   logistic-regression tuning occur within the training folds only.
5. Select hyperparameters by inner ROC-AUC and report outer-fold and pooled
   out-of-fold metrics at a fixed 0.5 threshold.

Metrics include accuracy, balanced accuracy, sensitivity, specificity, precision,
death-class F1, ROC-AUC, and average precision (the original code calls this PR_AUC).
Overlapping tokens contribute more than once during aggregation. Pooled OOF AUC
mixes separately fitted fold models; inspect fold metrics as well. Selecting an
encoder using these outer scores requires a further independent test set.

## Run

Use Python 3.10+ in a dedicated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --csv /approved/local/cohort.csv --output-dir artifacts/run01 --confirm-pre-outcome-notes
```

CSV columns: `patient_id`, `note`, `death_event`. Labels must be independently
reviewed: 1 means recorded death, 0 means no recorded death. Unlike the notebook,
this sample does not infer labels from arbitrary nonempty death-date strings or
rename a tumor-size column into notes. At least ten patients per class are required.
The flag confirms manual review that notes were available at the intended prediction
time; it does not automatically establish temporal validity or remove outcome leakage.

CPU is the default. Model/tokenizer downloads contact Hugging Face; notes are
processed locally, not sent to an inference API. Review the model's access and
license conditions separately. Dependency ranges are not an environment lockfile.

## Privacy and limitations

Original notebooks, outputs, dates, notes, datasets, identifiers, cached embeddings,
patient predictions, and fitted classifiers are excluded. Only aggregate metrics
and configuration are written locally; even aggregate results require governance
review before sharing. `.gitignore` is not a security control. Original files are
unchanged. No license is assigned to collaborators' underlying research or data.

Patient uniqueness and text-duplicate checks do not establish external validity.
Follow-up completeness, outcome adjudication, note timestamps, calibration,
subgroup fairness, confidence intervals, and external validation remain necessary.
No full model download, clinical training, or clinical reproduction was performed
when preparing this code. This is a research sample, not a clinical decision tool.
