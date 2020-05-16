import torch
import torch.nn as nn
import torch.nn.functional as F

import datetime
import time
import logging
import pandas as pd
import numpy as np
import os

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from tqdm import tqdm

class TimeTracker(object):

    def __init__(self, length=100):
        self.length = length
        self.load_time = []
        self.step_time = []

    def set_time(self, t):
        self.load_time.append(t[0])
        self.step_time.append(t[1])

    def get_time(self):
        return (np.mean(self.load_time[-int(self.length):]),
                np.mean(self.step_time[-int(self.length):]))

class LossTracker(object): 

    def __init__(self, num_moving_average=1000): 
        self.losses = []
        self.history = []
        self.avg = num_moving_average

    def set_loss(self, minibatch_loss): 
        self.losses.append(minibatch_loss) 

    def get_loss(self): 
        self.history.append(np.mean(self.losses[-self.avg:]))
        return self.history[-1]

    def reset(self): 
        self.losses = [] 

    def get_history(self): 
        return self.history

class Trainer(object):

    def __init__(self, 
        loader,
        model, 
        optimizer,
        schedule, 
        criterion, 
        evaluator):

        self.model = model 
        self.loader = loader
        self.optimizer = optimizer
        self.scheduler = schedule
        self.criterion = criterion
        self.evaluator = evaluator

        self.loss_tracker = LossTracker(num_moving_average=1000)
        self.time_tracker = TimeTracker(length=100)
        
        self.logfile = os.path.join(self.evaluator.save_checkpoint_dir, 'log.txt')
        self.init_logger()

    def init_logger(self):
        if os.path.exists(self.logfile): os.system('rm {}'.format(self.logfile))
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger()
        self.logger.addHandler(logging.FileHandler(self.logfile, 'a'))
        self.print = self.logger.info
        self.evaluator.set_logger(self.logger)

    def check_end_train(self): 
        return self.current_epoch >= self.num_epochs

    def check_end_epoch(self):
        return (self.steps % self.steps_per_epoch) == 0 and (self.steps > 0)

    def check_validation(self):
        # We add 1 to current_epoch when checking whether to validate
        # because epochs are 0-indexed. E.g., if validate_interval is 2,
        # we should validate after epoch 1. We need to add 1 so the modulo
        # returns 0
        return self.check_end_epoch() and self.steps > 0 and ((self.current_epoch + 1) % self.validate_interval) == 0

    def scheduler_step(self):
        if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
            self.scheduler.step(self.current_epoch + self.steps * 1./self.steps_per_epoch)
        else:
            self.scheduler.step()

    # Move the model forward ...
    def _fetch_output(self, data): 
        batch, labels, time = data 
        return (self.model(batch), batch, labels, time)

    def _step(self, data):
        output, batch, labels, time = self._fetch_output(data)

        self.loss = self.criterion(output, labels)
        self.loss.backward() 
        self.loss_tracker.set_loss(self.loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _accumulate_step(self, data):
        output, batch, labels, time = self._fetch_output(data)

        self.loss = self.criterion(output, labels)
        self.tracker_loss += self.loss.item()

        if self.grad_iter < (self.gradient_accumulation - 1):
            retain = True
        else:
            retain = False
        (self.loss / self.gradient_accumulation).backward(retain_graph=retain) 
        self.grad_iter += 1

        if self.grad_iter == self.gradient_accumulation:
            # Once the number of iterations has been reached
            # Step forward and reset
            self.loss_tracker.set_loss(self.tracker_loss / self.gradient_accumulation)
            self.optimizer.step()
            self.optimizer.zero_grad() 
            self.tracker_loss = 0.

    def train_step(self, data):
        self._accumulate_step(data) if self.gradient_accumulation > 1 else self._step(data)

    def complete_step(self, times):
        data_time, step_time = times
        self.step_complete = True
        self.grad_iter = 0
        self.steps += 1
        self.time_tracker.set_time((data_time, step_time))
        if self.scheduler.update == 'on_batch':
            self.scheduler_step()
    #

    def print_progress(self):
        self.print('epoch {epoch}, batch {batch} / {steps_per_epoch} : loss = {train_loss:.4f} (data: {load_time:.3f} sec/batch, step: {step_time:.3f} sec/batch)'
                .format(epoch=str(self.current_epoch).zfill(len(str(self.num_epochs))), \
                        batch=str(self.steps).zfill(len(str(self.steps_per_epoch))), \
                        steps_per_epoch=self.steps_per_epoch, \
                        train_loss=self.loss_tracker.get_loss(), \
                        load_time=self.time_tracker.get_time()[0],
                        step_time=self.time_tracker.get_time()[1]))

    def init_training(self, 
                      gradient_accumulation, 
                      num_epochs,
                      steps_per_epoch,
                      validate_interval):

        self.gradient_accumulation = float(gradient_accumulation)
        self.num_epochs = num_epochs
        self.steps_per_epoch = len(self.loader) if steps_per_epoch == 0 else steps_per_epoch
        self.validate_interval = validate_interval

        self.steps = 0 
        self.step_complete = True
        self.current_epoch = 0
        self.grad_iter = 0
        self.tracker_loss = 0

        self.optimizer.zero_grad()

    def train(self, 
              gradient_accumulation,
              num_epochs, 
              steps_per_epoch, 
              validate_interval,
              verbosity=100): 
        # Epochs are 0-indexed
        self.init_training(gradient_accumulation, num_epochs, steps_per_epoch, validate_interval)
        start_time = datetime.datetime.now()
        while 1: 
            for i, data in enumerate(self.loader):
                # Data loader should always return data load time as last element in tuple
                if self.step_complete:
                    step_time = 0.
                    data_time = 0.
                self.step_complete = False
                # GRADIENT ACCUMULATION
                if self.gradient_accumulation > 1:
                    step_start = time.time()
                    self.train_step(data)
                    step_time += time.time() - step_start
                    data_time += data[-1].detach().cpu().numpy().sum()
                    # Step is not complete until gradient accumulation is finished
                    if self.grad_iter == self.gradient_accumulation:
                        self.complete_step((data_time, step_time))
                # STANDARD
                else:
                    step_start = time.time()
                    self.train_step(data)
                    step_time += time.time() - step_start
                    data_time += data[-1].detach().cpu().numpy().sum()
                    self.complete_step((data_time, step_time))

                # Check- print training progress
                if self.steps % verbosity == 0 and self.steps > 0 and self.step_complete:
                    self.print_progress()
                # Check- run validation
                if self.check_validation():
                    self.print('VALIDATING ...')
                    validation_start_time = datetime.datetime.now()
                    # Start validation
                    self.model.eval()
                    valid_metric = self.evaluator.validate(self.model, 
                        self.criterion, 
                        str(self.current_epoch).zfill(len(str(self.num_epochs))))
                    if self.scheduler.update == 'on_valid':
                        self.scheduler.step(valid_metric)
                    # End validation
                    self.model.train()
                    self.print('Validation took {} !'.format(datetime.datetime.now() - validation_start_time))
                # Check- end of epoch
                if self.check_end_epoch():
                    if self.scheduler.update == 'on_epoch':
                        self.scheduler.step()
                    self.current_epoch += 1
                    self.steps = 0
                    # RESET BEST MODEL IF USING COSINEANNEALINGWARMRESTARTS
                    if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                        if self.current_epoch % self.scheduler.T_0 == 0:
                            self.evaluator.reset_best()
                #
                if self.evaluator.check_stopping(): 
                    # Make sure to set number of epochs to max epochs
                    # Remember, epochs are 0-indexed and we added 1 already
                    # So, this should work (e.g., epoch 99 would now be epoch 100,
                    # thus training would stop after epoch 99 if num_epochs = 100)
                    self.current_epoch = num_epochs
                if self.check_end_train():
                    # Break the for loop
                    break
            if self.check_end_train(): 
                # Break the while loop
                break 
        self.print('TRAINING : END') 
        self.print('Training took {}\n'.format(datetime.datetime.now() - start_time))








