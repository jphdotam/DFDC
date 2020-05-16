import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

class MixupBCELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        if type(y_true) == dict:
            # Training
            y_true1 = y_true['y_true1']
            y_true2 = y_true['y_true2']
            lam = y_true['lam']
            mix_loss1 = F.binary_cross_entropy_with_logits(y_pred, y_true1, reduction='none')
            mix_loss2 = F.binary_cross_entropy_with_logits(y_pred, y_true2, reduction='none')
            return (lam * mix_loss1 + (1. - lam) * mix_loss2).mean()
        else:
            # Validation
            return F.binary_cross_entropy_with_logits(y_pred, y_true)


class AugMixBCE(nn.Module):

    def __init__(self, lam=12):
        super().__init__()
        self.lam = lam

    def forward_train(self, y_pred, y_true):
        y_pred_orig, y_pred_aug1, y_pred_aug2 = y_pred['orig'], y_pred['aug1'], y_pred['aug2']
        # Compute loss on clean images
        loss = F.binary_cross_entropy_with_logits(y_pred_orig, y_true)
        p_orig, p_aug1, p_aug2 = torch.sigmoid(y_pred_orig), \
                                 torch.sigmoid(y_pred_aug1), \
                                 torch.sigmoid(y_pred_aug2)
        # Clamp mixture distribution to avoid exploding KL divergence
        p_mixture = torch.clamp((p_orig + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
        jsd = (F.kl_div(p_mixture, p_orig, reduction='batchmean') +
               F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
               F.kl_div(p_mixture, p_aug2, reduction='batchmean'))
        jsd /= 3.
        loss += self.lam * jsd
        return loss

    def forward(self, y_pred, y_true):
        if type(y_pred) == dict:
            return self.forward_train(y_pred, y_true)
        else:
            # Validation
            return F.binary_cross_entropy_with_logits(y_pred, y_true)


class HybridSegClsLoss(nn.Module):

    def __init__(self, seg_weight=100, seg_loss='mse_loss'):
        super().__init__()
        self.seg_weight = seg_weight
        self.seg_loss = seg_loss

    def forward_train(self, y_pred, y_true):
        logits, segmentation = y_pred
        y_true_cls, y_true_seg = y_true['cls'].float(), y_true['seg']
        seg_loss = getattr(F, self.seg_loss)(segmentation, y_true_seg, reduction='none')
        if seg_loss.ndim == 4:
            seg_loss = seg_loss.mean((-1,-2,-3)).mean() 
        elif seg_loss.ndim == 3:
            seg_loss = seg_loss.mean((-1,-2)).mean()
        cls_loss = F.binary_cross_entropy_with_logits(logits, y_true_cls)
        seg_loss *= self.seg_weight
        return cls_loss + seg_loss

    def forward_test(self, y_pred, y_true):
        return F.binary_cross_entropy_with_logits(y_pred, y_true)

    def forward(self, y_pred, y_true):
        if type(y_pred) == tuple and len(y_pred) == 2:
            return self.forward_train(y_pred, y_true)
        else:
            return self.forward_test(y_pred, y_true['cls'].float())


class MixupHybridSegClsLoss(nn.Module):

    def __init__(self, seg_weight=100, seg_loss='mse_loss'):
        super().__init__()
        self.seg_weight = seg_weight
        self.seg_loss = seg_loss

    def forward_train(self, y_pred, y_true):
        logits, segmentation = y_pred
        # y_true will be a dict with keys: 
        #  - y_true_seg, y_true1, y_true2, lam
        y_true_seg = y_true['y_true_seg']
        y_true_cls1 = y_true['y_true1']
        y_true_cls2 = y_true['y_true2']
        lam = y_true['lam']
        seg_loss = getattr(F, self.seg_loss)(segmentation, y_true_seg, reduction='none')
        if seg_loss.ndim == 4:
            seg_loss = seg_loss.mean((-1,-2,-3)).mean() 
        elif seg_loss.ndim == 3:
            seg_loss = seg_loss.mean((-1,-2)).mean()
        cls_loss1 = F.binary_cross_entropy_with_logits(logits, y_true_cls1.float(), reduction='none')
        cls_loss2 = F.binary_cross_entropy_with_logits(logits, y_true_cls2.float(), reduction='none')        
        cls_loss = (lam * cls_loss1 + (1. - lam) * cls_loss2).mean()
        seg_loss *= self.seg_weight
        return cls_loss + seg_loss

    def forward_test(self, y_pred, y_true):
        return F.binary_cross_entropy_with_logits(y_pred, y_true)

    def forward(self, y_pred, y_true):
        if type(y_pred) == tuple and len(y_pred) == 2:
            return self.forward_train(y_pred, y_true)
        else:
            return self.forward_test(y_pred, y_true['cls'].float())


class OHEMMixupBCELoss(MixupBCELoss):

    def __init__(self, total_steps, lowest_rate=1./8):
        super().__init__()
        self.total_steps = total_steps
        self.lowest_rate = lowest_rate
        self.steps = 0

    def _annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def calculate_rate(self):
        pct = float(self.steps) / self.total_steps
        self.steps += 1
        self.current_rate = self._annealing_cos(start=1.0, end=self.lowest_rate, pct=pct)

    def forward_test(self, y_pred, y_true, reduction='mean'):
        return F.binary_cross_entropy_with_logits(y_pred, y_true)

    def forward_mix(self, y_pred, y_true):
        y_true1 = y_true['y_true1']
        y_true2 = y_true['y_true2']
        lam = y_true['lam']
        mix_loss1 = F.binary_cross_entropy_with_logits(y_pred, y_true1, reduction='none')
        mix_loss2 = F.binary_cross_entropy_with_logits(y_pred, y_true2, reduction='none')
        return (lam * mix_loss1 + (1. - lam) * mix_loss2)

    def forward(self, y_pred, y_true):
        if type(y_true) == dict:
            # Training
            loss = self.forward_mix(y_pred, y_true)
            B = y_pred.size(0)
            self.calculate_rate()
            loss, _ = loss.topk(k=int(self.current_rate * B))
            return loss.mean()
        else:
            # Validation
            return self.forward_test(y_pred, y_true)
