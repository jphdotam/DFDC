import argparse
import logging
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yaml
import copy
import re
import os, os.path as osp

from tqdm import tqdm

try:
    from .factory import set_reproducibility
    from .factory import train as factory_train
    from .factory import evaluate as factory_evaluate
    from .factory import builder 
except:
    from factory import set_reproducibility
    import factory.train as factory_train
    import factory.evaluate as factory_evaluate
    import factory.builder as builder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('mode', type=str) 
    parser.add_argument('--gpu', type=lambda s: [int(_) for _ in s.split(',')] , default=[0])
    parser.add_argument('--num-workers', type=int, default=-1)
    return parser.parse_args()

def create_logger(cfg, mode):
    logfile = osp.join(cfg['evaluation']['params']['save_checkpoint_dir'], 'log_{}.txt'.format(mode))
    if osp.exists(logfile): os.system('rm {}'.format(logfile))
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('root')
    logger.addHandler(logging.FileHandler(logfile, 'a'))
    return logger

def set_inference_batch_size(cfg):
    if 'evaluation' in cfg.keys() and 'train' in cfg.keys():
        if 'batch_size' not in cfg['evaluation'].keys(): 
            cfg['evaluation']['batch_size'] = 2*cfg['train']['batch_size']

    if 'test' in cfg.keys() and 'train' in cfg.keys():
        if 'batch_size' not in cfg['test'].keys(): 
            cfg['test']['batch_size'] = 2*cfg['train']['batch_size']

    if 'predict' in cfg.keys() and 'train' in cfg.keys():
        if 'batch_size' not in cfg['predict'].keys(): 
            cfg['predict']['batch_size'] = 2*cfg['train']['batch_size']

    return cfg 

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    if args.num_workers > 0:
        if 'transform' not in cfg.keys():
            cfg['transform'] = {}
        cfg['transform']['num_workers'] = args.num_workers

    cfg = set_inference_batch_size(cfg)

    # We will set all the seeds we can, in vain ...
    set_reproducibility(cfg['seed'])
    # Set GPU
    if len(args.gpu) == 1:
        torch.cuda.set_device(args.gpu[0])

    if args.mode == 'predict':
        predict(args, cfg)
        return

    if 'mixup' not in cfg['train']['params'].keys():
        cfg['train']['params']['mixup'] = None

    if 'cutmix' not in cfg['train']['params'].keys():
        cfg['train']['params']['cutmix'] = None

    # Make directory to save checkpoints
    if not osp.exists(cfg['evaluation']['params']['save_checkpoint_dir']): 
        os.makedirs(cfg['evaluation']['params']['save_checkpoint_dir'])

    # Load in labels with CV splits
    df = pd.read_csv(cfg['dataset']['csv_filename'])
    ofold = cfg['dataset']['outer_fold']
    ifold = cfg['dataset']['inner_fold']

    train_df, valid_df, test_df = get_train_valid_test(cfg, df, ofold, ifold)

    logger = create_logger(cfg, args.mode)
    logger.info('Saving to {} ...'.format(cfg['evaluation']['params']['save_checkpoint_dir']))

    if args.mode == 'find_lr':
        cfg['optimizer']['params']['lr'] = cfg['find_lr']['params']['start_lr']
        find_lr(args, cfg, train_df, valid_df)
    elif args.mode == 'train':
        train(args, cfg, train_df, valid_df)
    elif args.mode == 'test':
        test(args, cfg, test_df)

def get_train_valid_test(cfg, df, ofold, ifold):
    train_df = df[df['split'] == 'train']
    valid_df = df[df['split'] == 'valid']
    test_df  = df[df['split'] == 'valid']
    return train_df, valid_df, test_df

def get_invfreq_weights(values, scale=None):
    logger = logging.getLogger('root')
    values, counts = np.unique(values, return_counts=True)
    num_samples = np.sum(counts)
    freqs = counts / float(num_samples)
    max_freq = np.max(freqs)
    invfreqs = max_freq / freqs
    if scale == 'log':
        logger.info('  Log scaling ...') 
        invfreqs = np.log(invfreqs+1)
    elif scale == 'sqrt':
        logger.info('  Square-root scaling ...')
        invfreqs = np.sqrt(invfreqs)
    invfreqs = invfreqs / np.sum(invfreqs)
    return invfreqs

def get_faceseq_labels(df, cfg):
    df.loc[:,'frame_num'] = [int(_.split('/')[-1].split('-')[0].replace('FRAME', '')) for _ in df['imgfile']]
    df.loc[:,'face_num']  = [int(_.split('/')[-1].split('-')[1].replace('FACE', '').replace('.png', '')) for _ in df['imgfile']]
    df = df.sort_values(['filename', 'frame_num', 'face_num']).reset_index(drop=True)
    df.loc[:,'filename_face'] = df['filename'] + df['face_num'].astype('str')
    images = []
    labels = []
    for fi, _df in tqdm(df.groupby('filename_face'), total=len(df['filename_face'].unique())):
        images.append([osp.join(cfg['dataset']['data_dir'], _) for _ in _df['imgfile']])
        assert len(_df['label'].unique()) == 1
        labels.append(_df['label'].iloc[0])
    return images, labels

def get_stacked_labels(df, cfg):
    N = cfg['dataset']['params']['stack']
    if N == 'diff':
        N = 2
    df.loc[:,'frame_num'] = [int(_.split('/')[-1].split('-')[0].replace('FRAME', '')) for _ in df['imgfile']]
    df.loc[:,'face_num']  = [int(_.split('/')[-1].split('-')[1].replace('FACE', '').replace('.png', '')) for _ in df['imgfile']]
    df = df.sort_values(['filename', 'frame_num', 'face_num']).reset_index(drop=True)
    df.loc[:,'filename_face'] = df['filename'] + df['face_num'].astype('str')
    images = []
    labels = []
    new_df = []
    for fi, _df in tqdm(df.groupby('filename_face'), total=len(df['filename_face'].unique())):
        assert len(_df['label'].unique()) == 1
        for i in range(_df.shape[0] - N + 1):
            images.append([osp.join(cfg['dataset']['data_dir'], _) for _ in list(_df['imgfile'])[i:N+i]])
            labels.append(_df['label'].iloc[0])
        new_df.append(_df.iloc[:(_df.shape[0] - N + 1)])
    new_df = pd.concat(new_df).reset_index(drop=True)
    assert new_df.shape[0] == len(images) == len(labels)
    return images, labels, new_df

def setup(args, cfg, train_df, valid_df):

    logger = logging.getLogger('root')

    if cfg['dataset']['name'] == 'FaceMaskDataset':
        train_images = [osp.join(cfg['dataset']['data_dir'], '{}'.format(_)) for _ in train_df['imgfile']]
        valid_images = [osp.join(cfg['dataset']['data_dir'], '{}'.format(_)) for _ in valid_df['imgfile']]
        train_labels = np.asarray(train_df['label'])
        valid_labels = np.asarray(valid_df['label'])
        train_masks  = [osp.join(cfg['dataset']['data_dir'], '{}'.format(_)) for _ in train_df['maskfile']]
        valid_masks  = [osp.join(cfg['dataset']['data_dir'], '{}'.format(_)) for _ in valid_df['maskfile']]

        train_loader = builder.build_dataloader(cfg, data_info={'imgfiles': train_images, 'maskfiles': train_masks, 'labels': train_labels}, mode='train')
        valid_loader = builder.build_dataloader(cfg, data_info={'imgfiles': valid_images, 'maskfiles': valid_masks, 'labels': valid_labels}, mode='valid')

    else:
        train_images = [osp.join(cfg['dataset']['data_dir'], '{}'.format(_)) for _ in train_df['vidfile']]
        valid_images = [osp.join(cfg['dataset']['data_dir'], '{}'.format(_)) for _ in valid_df['vidfile']]
        train_labels = np.asarray(train_df['label'])
        valid_labels = np.asarray(valid_df['label'])

        train_loader = builder.build_dataloader(cfg, data_info={'vidfiles': train_images, 'labels': train_labels}, mode='train')
        valid_loader = builder.build_dataloader(cfg, data_info={'vidfiles': valid_images, 'labels': valid_labels}, mode='valid')

    # Adjust steps per epoch if necessary (i.e., equal to 0)
    # We assume if gradient accumulation is specified, then the user
    # has already adjusted the steps_per_epoch accordingly in the 
    # config file
    steps_per_epoch = cfg['train']['params']['steps_per_epoch']
    gradient_accmul = cfg['train']['params']['gradient_accumulation']
    if steps_per_epoch == 0:
        cfg['train']['params']['steps_per_epoch'] = len(train_loader)
        # if gradient_accmul > 1:
        #     new_steps_per_epoch = int(cfg['train']['params']['steps_per_epoch'] 
        #                               / gradient_accmul)
        #     cfg['train']['params']['steps_per_epoch'] = new_steps_per_epoch
    

    # Generic build function will work for model/loss
    logger.info('Building [{}] architecture ...'.format(cfg['model']['name']))
    if 'backbone' in cfg['model']['params'].keys():
        logger.info('  Using [{}] backbone ...'.format(cfg['model']['params']['backbone']))
    if 'pretrained' in cfg['model']['params'].keys():
        logger.info('  Pretrained weights : {}'.format(cfg['model']['params']['pretrained']))
    model = builder.build_model(cfg['model']['name'], cfg['model']['params'])
    model = model.train().cuda()

    if cfg['loss']['params'] is None:
        cfg['loss']['params'] = {}

    if re.search(r'^OHEM', cfg['loss']['name']):
        cfg['loss']['params']['total_steps'] = cfg['train']['params']['num_epochs'] * cfg['train']['params']['steps_per_epoch']

    criterion = builder.build_loss(cfg['loss']['name'], cfg['loss']['params'])
    optimizer = builder.build_optimizer(
        cfg['optimizer']['name'], 
        model.parameters(), 
        cfg['optimizer']['params'])
    scheduler = builder.build_scheduler(
        cfg['scheduler']['name'], 
        optimizer, 
        cfg=cfg)

    if len(args.gpu) > 1:
        model = nn.DataParallel(model, device_ids=args.gpu)

    return cfg, \
           train_loader, \
           valid_loader, \
           model, \
           optimizer, \
           criterion, \
           scheduler 

def find_lr(args, cfg, train_df, valid_df):

    logger = logging.getLogger('root')

    logger.info('FINDING LR ...')

    cfg, \
    train_loader, \
    valid_loader, \
    model, \
    optimizer, \
    criterion, \
    scheduler = setup(args, cfg, train_df, valid_df)

    finder = factory_train.LRFinder(
        loader=train_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        save_checkpoint_dir=cfg['evaluation']['params']['save_checkpoint_dir'],
        logger=logger,
        gradient_accumulation=cfg['train']['params']['gradient_accumulation'],
        mixup=cfg['train']['params']['mixup'],
        cutmix=cfg['train']['params']['cutmix'])

    finder.find_lr(**cfg['find_lr']['params'])

    logger.info('Results are saved in : {}'.format(osp.join(finder.save_checkpoint_dir, 'lrfind.csv')))

def train(args, cfg, train_df, valid_df):
    
    logger = logging.getLogger('root')

    logger.info('TRAINING : START')

    if 'oneface' in cfg['dataset'].keys() and cfg['dataset']['oneface']:
        logger.info('Using only FAKE videos with one face ...')
        train_df = train_df[((train_df['oneface'] == 1) | (train_df['label'] == 0))]

    logger.info('TRAIN: n={}'.format(len(train_df)))
    logger.info('VALID: n={}'.format(len(valid_df)))

    cfg, \
    train_loader, \
    valid_loader, \
    model, \
    optimizer, \
    criterion, \
    scheduler = setup(args, cfg, train_df, valid_df)

    evaluator = getattr(factory_evaluate, cfg['evaluation']['evaluator'])
    evaluator = evaluator(loader=valid_loader,
        **cfg['evaluation']['params'])

    trainer = getattr(factory_train, cfg['train']['trainer'])
    trainer = trainer(loader=train_loader,
        model=model,
        optimizer=optimizer,
        schedule=scheduler,
        criterion=criterion,
        evaluator=evaluator,
        logger=logger)
    trainer.train(**cfg['train']['params'])


def test(args, cfg, test_df):

    if 'csv_filename' in cfg['test'].keys():
        if cfg['test']['csv_filename']:
            test_df = pd.read_csv(cfg['test']['csv_filename'])

    logger = logging.getLogger('root')
    logger.info('TESTING : START')
    logger.info('TEST: n={}'.format(len(test_df)))

    if 'data_dir' in cfg['test'].keys():
        if cfg['test']['data_dir']: 
            cfg['dataset']['data_dir'] = cfg['test']['data_dir']

    test_df = test_df[test_df['part'] != 45]
    test_images = [osp.join(cfg['dataset']['data_dir'], '{}'.format(_)) for _ in test_df['vidfile']]
    test_labels = np.asarray(test_df['label'])

    test_loader = builder.build_dataloader(cfg, data_info={'vidfiles': test_images, 'labels': test_labels}, mode='test')

    cfg['model']['params']['pretrained'] = None
    model = builder.build_model(cfg['model']['name'], cfg['model']['params'])
    model.load_state_dict(torch.load(cfg['test']['checkpoint'], map_location=lambda storage, loc: storage))
    model = model.eval().cuda()

    if 'params' not in cfg['test'].keys() or type(cfg['test']['params']) == type(None):
        cfg['test']['params'] = {}

    predictor = getattr(factory_evaluate, cfg['test']['predictor'])
    predictor = predictor(loader=test_loader,
        **cfg['test']['params'])

    y_true, y_pred, _ = predictor.predict(model, criterion=None, epoch=None)

    if not osp.exists(cfg['test']['save_preds_dir']):
        os.makedirs(cfg['test']['save_preds_dir'])

    with open(osp.join(cfg['test']['save_preds_dir'], 'predictions.pkl'), 'wb') as f:
        pickle.dump({
            'y_true': y_true,
            'y_pred': y_pred,
            'imgfiles': [im.split('/')[-1] for im in test_images]
        }, f)

def predict(args, cfg):

    logger = logging.getLogger('root')
    logger.info('PREDICT : START')

    if 'model_weights' not in cfg.keys() or cfg['model_weights'] is None:
        cfg['model_weights'] = [1.] * len(cfg['model_configs'])

    assert len(cfg['model_weights']) == len(cfg['model_configs'])

    if 'data_dir' in cfg['predict'].keys():
        if cfg['predict']['data_dir']: 
            cfg['dataset']['data_dir'] = cfg['predict']['data_dir']

    if type(cfg['predict']['path_to_parquet']) != list:
        cfg['predict']['path_to_parquet'] = list(cfg['predict']['path_to_parquet'])

    assert cfg['dataset']['name'] == 'BengaliParquetDataset'

    predict_parquets = [osp.join(cfg['dataset']['data_dir'], path_to_parquet)
        for path_to_parquet in cfg['predict']['path_to_parquet']
    ]

    model_configs = []
    for cfgfile in cfg['model_configs']:
        with open(cfgfile) as f:
            model_cfg = yaml.load(f, Loader=yaml.FullLoader)    
        model_cfg['model']['params']['pretrained'] = None
        if 'grapheme_model_checkpoint' in model_cfg['model']['params'].keys():
            model_cfg['model']['params']['grapheme_model_checkpoint'] = None
        model_configs.append(model_cfg)

    def create_model(cfg):
        model = builder.build_model(cfg['model']['name'], cfg['model']['params'])
        model.load_state_dict(torch.load(cfg['test']['checkpoint'], map_location=lambda storage, loc: storage))
        model = model.eval().cuda()
        return model

    models = [create_model(model_cfg) for model_cfg in model_configs]

    cfg['predict']['labels_available'] = False
    image_ids = []
    y_pred_list = []
    for parquet in predict_parquets:
        loader = builder.build_dataloader(cfg, data_info={'path_to_parquet': parquet}, mode='predict')
        image_ids.extend(list(loader.dataset.image_ids))
        predictor = getattr(factory_evaluate, cfg['predict']['predictor'])
        predictor = predictor(loader=loader, **cfg['predict']['params'])
        y_pred = []
        for m in models:
            _, single_y_pred, _ = predictor.predict(m, criterion=None, epoch=None)
            y_pred.append(single_y_pred)
        y_pred_list.append(y_pred)

    weights = np.asarray(cfg['model_weights']) ; weights /= weights.sum()
    averaged_pred_list = []
    for y_pred in y_pred_list: 
        averaged = copy.deepcopy(y_pred[0])
        for k,v in averaged.items():
            averaged[k] = v * weights[0]
        for ind, each_y_pred in enumerate(y_pred[1:]):
            for k,v in each_y_pred.items():
                averaged[k] += v * weights[ind+1]
        averaged_pred_list.append(averaged)

    combined_preds = averaged_pred_list[0]
    for pred in averaged_pred_list[1:]:
        for k in combined_preds.keys():
            combined_preds[k] = np.vstack((combined_preds[k], pred[k]))

    row_id = []
    target = []
    for ind, i in enumerate(image_ids):
        for k in combined_preds.keys():
            row_id.append('{}_{}'.format(i, k))
            target.append(np.argmax(combined_preds[k][ind]))

    submission = pd.DataFrame({'row_id': row_id, 'target': target})
    submission.to_csv(cfg['predict']['submission_csv'], index=False)

if __name__ == '__main__':
    main()












