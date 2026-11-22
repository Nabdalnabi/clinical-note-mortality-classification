"""Frozen biomedical encoder helpers adapted from the supplied notebook."""
import gc
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DEVICE = torch.device("cpu")
embedding_parameter_grid = {
    "pca__n_components": [8, 16, 32, 64],
    "classifier__C": [0.001, 0.01, 0.1, 1., 10., 100.],
    "classifier__class_weight": [None, "balanced"],
}

def mean_pool_hidden_states(hidden_states, attention_mask):
    """Mean-pool nonpadding token embeddings."""
    expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(hidden_states.dtype)
    summed_embeddings = (hidden_states * expanded_mask).sum(dim=1)
    token_counts = expanded_mask.sum(dim=1).clamp(min=1e-09)
    return summed_embeddings / token_counts

def embed_single_note(note, tokenizer, model, device, max_length, stride, chunk_batch_size):
    """
    Split one note into overlapping chunks and return one
    patient-level embedding.

    Longer chunks receive greater weight during aggregation.
    """
    encoded_note = tokenizer(note, add_special_tokens=True, truncation=True, max_length=max_length, stride=stride, return_overflowing_tokens=True, padding=False)
    input_id_chunks = encoded_note['input_ids']
    if len(input_id_chunks) > 0 and isinstance(input_id_chunks[0], int):
        encoded_note['input_ids'] = [encoded_note['input_ids']]
        if 'attention_mask' in encoded_note:
            encoded_note['attention_mask'] = [encoded_note['attention_mask']]
        if 'token_type_ids' in encoded_note:
            encoded_note['token_type_ids'] = [encoded_note['token_type_ids']]
    number_of_chunks = len(encoded_note['input_ids'])
    chunk_embeddings = []
    chunk_weights = []
    allowed_model_inputs = {'input_ids', 'attention_mask'}
    if model.config.model_type == 'bert':
        allowed_model_inputs.add('token_type_ids')
    with torch.no_grad():
        for chunk_start in range(0, number_of_chunks, chunk_batch_size):
            chunk_end = min(chunk_start + chunk_batch_size, number_of_chunks)
            chunk_features = []
            for chunk_index in range(chunk_start, chunk_end):
                one_chunk = {}
                for key in allowed_model_inputs:
                    if key in encoded_note:
                        one_chunk[key] = encoded_note[key][chunk_index]
                chunk_features.append(one_chunk)
            padded_batch = tokenizer.pad(chunk_features, padding=True, return_tensors='pt')
            model_inputs = {key: value.to(device) for (key, value) in padded_batch.items() if key in allowed_model_inputs}
            if model.config.model_type == 'longformer':
                global_mask = torch.zeros_like(model_inputs['attention_mask'])
                global_mask[:, 0] = 1
                model_inputs['global_attention_mask'] = global_mask
            model_outputs = model(**model_inputs)
            hidden_states = model_outputs.last_hidden_state
            pooled_chunks = mean_pool_hidden_states(hidden_states, model_inputs['attention_mask'])
            valid_token_counts = model_inputs['attention_mask'].sum(dim=1)
            chunk_embeddings.append(pooled_chunks.detach().cpu().numpy())
            chunk_weights.append(valid_token_counts.detach().cpu().numpy())
    chunk_embeddings = np.vstack(chunk_embeddings)
    chunk_weights = np.concatenate(chunk_weights).astype(float)
    patient_embedding = np.average(chunk_embeddings, axis=0, weights=chunk_weights)
    return patient_embedding.astype(np.float32)

def extract_patient_embeddings(notes, model_label, model_specification, device):
    """Extract one frozen encoder embedding per patient."""
    model_id = model_specification['model_id']
    print('\n' + '=' * 70)
    print('Loading:', model_label)
    print('Hugging Face model:', model_id)
    print('=' * 70)
    load_options = dict(revision=model_specification.get('revision', 'main'),
                        local_files_only=model_specification.get('local_files_only', False),
                        trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **load_options)
    model = AutoModel.from_pretrained(model_id, low_cpu_mem_usage=False, **load_options)
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    all_patient_embeddings = []
    for note in tqdm(notes, desc=f'Embedding with {model_label}'):
        patient_embedding = embed_single_note(note=note, tokenizer=tokenizer, model=model, device=device, max_length=model_specification['max_length'], stride=model_specification['stride'], chunk_batch_size=model_specification['chunk_batch_size'])
        all_patient_embeddings.append(patient_embedding)
    embedding_matrix = np.vstack(all_patient_embeddings)
    if not np.isfinite(embedding_matrix).all():
        raise ValueError('Encoder produced NaN or infinite embeddings.')
    print('\nEmbedding shape:', embedding_matrix.shape)
    del model
    del tokenizer
    del all_patient_embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, 'mps') and DEVICE.type == 'mps':
        torch.mps.empty_cache()
    return embedding_matrix

def create_embedding_classifier():
    """
    Standardize embeddings, reduce dimensionality and train
    regularized logistic regression.
    """
    return Pipeline(steps=[('scaler', StandardScaler()), ('pca', PCA(n_components=16, svd_solver='randomized', random_state=RANDOM_STATE)), ('classifier', LogisticRegression(penalty='l2', solver='liblinear', max_iter=5000, random_state=RANDOM_STATE))])

def evaluate_note_embeddings(embedding_matrix, outcome, encoder_name):
    """Evaluate frozen embeddings using nested five-fold CV."""
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_records = []
    fold_best_parameters = []
    oof_death_probability = np.full(len(outcome), np.nan)
    for (outer_fold, (train_index, test_index)) in enumerate(outer_cv.split(embedding_matrix, outcome), start=1):
        print(f'\n{encoder_name}, outer fold {outer_fold}/5')
        X_train = embedding_matrix[train_index]
        X_test = embedding_matrix[test_index]
        y_train = outcome[train_index]
        y_test = outcome[test_index]
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE + outer_fold)
        inner_search = GridSearchCV(estimator=create_embedding_classifier(), param_grid=embedding_parameter_grid, scoring={'ROC_AUC': 'roc_auc', 'PR_AUC': 'average_precision'}, refit='ROC_AUC', cv=inner_cv, n_jobs=-1, verbose=0, error_score='raise', return_train_score=False)
        inner_search.fit(X_train, y_train)
        optimized_model = inner_search.best_estimator_
        death_probability = optimized_model.predict_proba(X_test)[:, 1]
        death_prediction = (death_probability >= 0.5).astype(int)
        oof_death_probability[test_index] = death_probability
        fold_record = {'Encoder': encoder_name, 'Outer_Fold': outer_fold, 'ROC_AUC': roc_auc_score(y_test, death_probability), 'PR_AUC': average_precision_score(y_test, death_probability), 'Balanced_Accuracy': balanced_accuracy_score(y_test, death_prediction), 'Sensitivity': recall_score(y_test, death_prediction, pos_label=1, zero_division=0), 'Specificity': recall_score(y_test, death_prediction, pos_label=0, zero_division=0), 'Precision_Death': precision_score(y_test, death_prediction, pos_label=1, zero_division=0), 'F1_Death': f1_score(y_test, death_prediction, zero_division=0), 'Accuracy': accuracy_score(y_test, death_prediction), 'Test_Patients': len(test_index), 'Test_Deaths': int(y_test.sum()), 'Best_Inner_ROC_AUC': inner_search.best_score_}
        fold_records.append(fold_record)
        fold_best_parameters.append({'Encoder': encoder_name, 'Outer_Fold': outer_fold, **inner_search.best_params_})
        print('Inner ROC-AUC:', round(inner_search.best_score_, 3))
        print('Outer ROC-AUC:', round(fold_record['ROC_AUC'], 3))
    fold_results = pd.DataFrame(fold_records)
    best_parameters = pd.DataFrame(fold_best_parameters)
    oof_prediction = (oof_death_probability >= 0.5).astype(int)
    oof_results = {'Encoder': encoder_name, 'OOF_ROC_AUC': roc_auc_score(outcome, oof_death_probability), 'OOF_PR_AUC': average_precision_score(outcome, oof_death_probability), 'OOF_Balanced_Accuracy': balanced_accuracy_score(outcome, oof_prediction), 'OOF_Sensitivity': recall_score(outcome, oof_prediction, pos_label=1, zero_division=0), 'OOF_Specificity': recall_score(outcome, oof_prediction, pos_label=0, zero_division=0), 'OOF_Precision_Death': precision_score(outcome, oof_prediction, pos_label=1, zero_division=0), 'OOF_F1_Death': f1_score(outcome, oof_prediction, zero_division=0), 'OOF_Accuracy': accuracy_score(outcome, oof_prediction)}
    return {'fold_results': fold_results, 'best_parameters': best_parameters, 'oof_death_probability': oof_death_probability, 'oof_prediction': oof_prediction, 'oof_results': oof_results}
