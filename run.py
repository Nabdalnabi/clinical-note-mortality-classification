"""Evaluate notes locally; export aggregate metrics only."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--output-dir', default='artifacts')
    parser.add_argument('--confirm-pre-outcome-notes', action='store_true')
    args = parser.parse_args()
    if not args.confirm_pre_outcome_notes:
        parser.error('Review note timing before using --confirm-pre-outcome-notes.')

    import numpy as np
    import pandas as pd
    import modeling as m
    from validation import validate

    frame = pd.read_csv(args.csv, dtype={'patient_id': 'string'})
    notes, outcome = validate(frame)
    m.torch.manual_seed(m.RANDOM_STATE)
    np.random.seed(m.RANDOM_STATE)
    specification = dict(model_id='nlpie/bio-distilbert-uncased', max_length=512,
                         stride=128, chunk_batch_size=8)
    embeddings = m.extract_patient_embeddings(notes, 'BioDistilBERT', specification, m.DEVICE)
    # Bound PCA by the smallest actual inner-training fold, not total cohort size.
    outer = m.StratifiedKFold(5, shuffle=True, random_state=m.RANDOM_STATE)
    limit = embeddings.shape[1]
    for fold, (train, _) in enumerate(outer.split(embeddings, outcome), 1):
        inner = m.StratifiedKFold(4, shuffle=True, random_state=m.RANDOM_STATE + fold)
        limit = min(limit, *(len(a) for a, _ in inner.split(embeddings[train], outcome[train])))
    m.embedding_parameter_grid['pca__n_components'] = [n for n in (8, 16, 32, 64) if n <= limit] or [1]
    result = m.evaluate_note_embeddings(embeddings, outcome, 'BioDistilBERT')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    result['fold_results'].to_csv(output / 'fold_metrics.csv', index=False)
    result['best_parameters'].to_csv(output / 'fold_parameters.csv', index=False)
    (output / 'metrics.json').write_text(json.dumps(result['oof_results'], indent=2))
    (output / 'configuration.json').write_text(json.dumps(specification, indent=2))
    print('Aggregate results saved. No notes, embeddings, identifiers, or predictions exported.')


if __name__ == '__main__':
    main()
