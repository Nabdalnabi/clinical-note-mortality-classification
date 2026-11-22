"""Input contract: independently adjudicated labels and one row per patient."""
def validate(frame):
    required = {'patient_id', 'note', 'death_event'}
    if not required.issubset(frame.columns):
        raise ValueError('Required columns: patient_id, note, death_event.')
    if frame['patient_id'].isna().any() or frame['patient_id'].str.strip().eq('').any():
        raise ValueError('Patient IDs must be nonempty.')
    if frame['patient_id'].str.strip().duplicated().any():
        raise ValueError('Exactly one row per patient is required.')
    notes = frame['note'].astype('string').fillna('').str.replace(r'\s+', ' ', regex=True).str.strip()
    if notes.eq('').any() or notes.str.lower().duplicated().any():
        raise ValueError('Empty or duplicate notes require review.')
    if not frame['death_event'].isin([0, 1]).all():
        raise ValueError('death_event must contain only adjudicated 0/1 labels.')
    counts = frame['death_event'].value_counts()
    if len(counts) != 2 or counts.min() < 10:
        raise ValueError('At least 10 patients per class are required for nested CV.')
    return notes.tolist(), frame['death_event'].to_numpy(dtype=int)
