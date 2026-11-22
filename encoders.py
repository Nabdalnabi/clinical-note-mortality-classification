"""Public checkpoint presets; not a claim of exact study reproduction."""
ENCODERS = {
    'biodistilbert': ('BioDistilBERT', 'nlpie/bio-distilbert-uncased', 512, 128, 8),
    'bioclinicalbert': ('BioClinicalBERT', 'emilyalsentzer/Bio_ClinicalBERT', 512, 128, 8),
    'biomedbert': ('BiomedBERT', 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext', 512, 128, 8),
    'clinical-longformer': ('Clinical-Longformer', 'yikuan8/Clinical-Longformer', 4096, 512, 1),
}


def specification(name):
    label, checkpoint, length, stride, batch = ENCODERS[name]
    return label, dict(model_id=checkpoint, max_length=length, stride=stride,
                      chunk_batch_size=batch)
