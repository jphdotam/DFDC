import torch
import pandas as pd
import numpy as np
import os, os.path as osp

from tqdm import tqdm
from pathlib import Path

from .metrics import *

from ..models import *
from ..data import cudaify, FaceMaskDataset

_PATH = Path(__file__).parent


class Predictor(object):

    def __init__(self,
                 loader,
                 labels_available=True,
                 cuda=True,
                 debug=False):

        self.loader = loader
        self.labels_available = labels_available
        self.cuda = cuda
        self.debug = debug

    def predict(self, model, criterion, epoch):
        self.epoch = epoch
        y_pred = []
        y_true = []   
        with torch.no_grad():
            losses = []
            for data in tqdm(self.loader, total=len(self.loader)):
                if self.debug:
                    if len(y_true) >= 200:
                        y_true[0] = 0
                        break
                batch, labels = data
                if self.cuda:
                    batch, labels = cudaify(batch, labels)
                output = model(batch)
                if criterion:
                    losses.append(criterion(output, labels).item())
                # Make sure you're using the right final transformation ...
                # softmax vs. sigmoid
                # if type(model) not in (SingleHead, SingleHeadX3, DiffModel):
                #     output = torch.sigmoid(output)
                if type(self.loader.dataset) == FaceMaskDataset:
                    labels = labels['cls']
                y_pred.extend(output.cpu().numpy())
                if self.labels_available:
                    y_true.extend(labels.cpu().numpy())
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        if type(self.loader.dataset) == FaceMaskDataset:
            names  = np.asarray(self.loader.dataset.videos[:len(y_true)])
            video_dict = {}
            for n in np.unique(names):
                video_dict[n] = np.where(names == n)[0]
            video_pred = np.asarray([np.mean(y_pred[video_dict[n]]) for n in np.unique(names)])
            video_true = np.asarray([np.mean(y_true[video_dict[n]]) for n in np.unique(names)])
            return video_true, video_pred, losses
        return y_true, y_pred, losses


class Evaluator(Predictor):

    def __init__(self,
                 loader,
                 metrics,
                 valid_metric,
                 mode,
                 improve_thresh,
                 prefix,
                 save_checkpoint_dir,
                 save_best,
                 early_stopping=np.inf,
                 thresholds=np.arange(0.05, 0.95, 0.05),
                 cuda=True,
                 debug=False):
        
        super(Evaluator, self).__init__(
            loader=loader, 
            cuda=cuda,
            debug=debug)

        if type(metrics) is not list: metrics = list(metrics)
        assert valid_metric in metrics

        self.loader = loader
        # List of strings corresponding to desired metrics
        # These strings should correspond to function names defined
        # in metrics.py
        self.metrics = metrics
        # valid_metric should be included within metrics
        # This specifies which metric we should track for validation improvement
        self.valid_metric = valid_metric
        # Mode should be one of ['min', 'max']
        # This determines whether a lower (min) or higher (max) 
        # valid_metric is considered to be better
        self.mode = mode
        # This determines by how much the valid_metric needs to improve
        # to be considered an improvement
        self.improve_thresh = improve_thresh
        # Specifies part of the model name
        self.prefix = prefix
        self.save_checkpoint_dir = save_checkpoint_dir
        # save_best = True, overwrite checkpoints if score improves
        # If False, save all checkpoints
        self.save_best = save_best
        self.metrics_file = os.path.join(save_checkpoint_dir, 'metrics.csv')
        if os.path.exists(self.metrics_file): os.system('rm {}'.format(self.metrics_file))
        # How many epochs of no improvement do we wait before stopping training?
        self.early_stopping = early_stopping
        self.stopping = 0
        self.thresholds = thresholds

        self.history = []
        self.epoch = None

        self.reset_best()

    def reset_best(self):
        self.best_model = None
        self.best_score = -np.inf

    def set_logger(self, logger):
        self.logger = logger
        self.print  = self.logger.info

    def validate(self, model, criterion, epoch):
        y_true, y_pred, losses = self.predict(model, criterion, epoch)
        valid_metric = self.calculate_metrics(y_true, y_pred, losses)
        self.save_checkpoint(model, valid_metric)
        return valid_metric

    def generate_metrics_df(self):
        df = pd.concat([pd.DataFrame(d, index=[0]) for d in self.history])
        df.to_csv(self.metrics_file, index=False)

    # Used by Trainer class
    def check_stopping(self):
        return self.stopping >= self.early_stopping

    def check_improvement(self, score):
        # If mode is 'min', make score negative
        # Then, higher score is better (i.e., -0.01 > -0.02)
        score = -score if self.mode == 'min' else score
        improved = score >= (self.best_score + self.improve_thresh)
        if improved:
            self.stopping = 0
        else:
            self.stopping += 1
        return improved

    def save_checkpoint(self, model, valid_metric):
        save_file = '{}_{}_VM-{:.4f}.pth'.format(self.prefix, str(self.epoch).zfill(3), valid_metric).upper()
        save_file = os.path.join(self.save_checkpoint_dir, save_file)
        if self.save_best:
            if self.check_improvement(valid_metric):
                if self.best_model is not None: 
                    os.system('rm {}'.format(self.best_model))
                self.best_model = save_file
                self.best_score = -valid_metric if self.mode == 'min' else valid_metric
                torch.save(model.state_dict(), save_file)
        else:
            torch.save(model.state_dict(), save_file)

    def calculate_metrics(self, y_true, y_pred, losses):
        metrics_dict = {}
        metrics_dict['loss'] = np.mean(losses)
        for metric in self.metrics:
            metric = eval(metric)
            metrics_dict.update(metric(y_true, y_pred, thresholds=self.thresholds))
        print_results = 'epoch {epoch} // VALIDATION'.format(epoch=self.epoch)
        max_str_len = np.max([len(k) for k in metrics_dict.keys()])
        for key in metrics_dict.keys():
            self.print('{key} | {value:.5g}'.format(key=key.ljust(max_str_len), value=metrics_dict[key]))
        valid_metric = metrics_dict[self.valid_metric]
        metrics_dict.update({'vm': valid_metric, 'epoch': int(self.epoch)})
        self.history.append(metrics_dict)
        self.generate_metrics_df()
        return valid_metric


