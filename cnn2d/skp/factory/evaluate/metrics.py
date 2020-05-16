from sklearn.metrics import log_loss as _log_loss
from sklearn.metrics import roc_auc_score, accuracy_score

import numpy as np

# dict key should match name of function

def log_loss(y_true, y_prob, **kwargs):
    return {'log_loss': _log_loss(y_true, y_prob, eps=1e-7)}

def auc(y_true, y_prob, **kwargs):
    return {'auc': roc_auc_score(y_true, y_prob)}

def accuracy(y_true, y_prob, **kwargs):
    return {'accuracy': accuracy_score(y_true, (y_prob >= 0.5).astype('float32'))}