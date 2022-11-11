import os
import sys
import time
import yaml
import wandb
import argparse

import pandas as pd
import numpy as np
# from tabulate import tabulate
from sklearn.model_selection import StratifiedKFold

import torch

from src import BC_Dataset, train_get_transforms, valid_get_transforms
from src import pl_Wrapper

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers


from src import obj, seed_everything, pl_Wrapper

# warnings
import warnings
warnings.filterwarnings('ignore')

# class config:
#     bs = args.BATCH_SIZE

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main(config):
    df_train = pd.read_csv('open/train.csv')
    df_test = pd.read_csv('open/test.csv')
    sub = pd.read_csv('open/sample_submission.csv')

    df_train['img_path'] = df_train['img_path'].apply(lambda x: x.replace('./', 'open/'))
    df_test['img_path'] = df_test['img_path'].apply(lambda x: x.replace('./', 'open/'))

    # preprocess outlier
    df_train['PR_Allred_score'] = df_train['PR_Allred_score'].where((0<=df_train['PR_Allred_score']) & (df_train['PR_Allred_score']<=8))

    for col in ['NG', 'HG', 'HG_score_1', 'HG_score_2', 'HG_score_3', 'DCIS_or_LCIS_type', 'ER_Allred_score', 'PR_Allred_score', 'HER2_SISH_ratio']:
        df_train[col].fillna(0, inplace=True)
        df_test[col].fillna(0, inplace=True)

    df_test['암의 장경'].fillna(df_train['암의 장경'].median(), inplace=True)
    df_train['암의 장경'].fillna(df_train['암의 장경'].median(), inplace=True)

    df_train['BRCA_mutation'] = df_train['BRCA_mutation'].fillna(1)
    df_test['BRCA_mutation'] = df_test['BRCA_mutation'].fillna(1)

    for col in ['T_category', 'HER2', 'HER2_IHC', 'HER2_SISH', 'KI-67_LI_percent']:
        df_train[col].fillna(-1, inplace=True)
        df_test[col].fillna(-1, inplace=True)

        df_train[col]+=1
        df_test[col]+=1

    df_train[['ER', 'PR']] = df_train[['ER', 'PR']].fillna(0)

    for col in config.train_params.cat_features:
        tmp_dict = {val:idx for idx, val in enumerate(np.unique(df_train[col]))}
        df_train[col] = df_train[col].map(tmp_dict)
        df_test[col] = df_test[col].map(tmp_dict)

    config.train_params.cat_features_ls = df_train[config.train_params.cat_features].nunique().values.tolist()
    config.train_params.num_numeric_features = len(config.train_params.numeric_features)
    config.embedding_size = 1024

    skf = StratifiedKFold(n_splits=config.train_params.folds, random_state=config.train_params.seed, shuffle=True)
    splits = list(skf.split(df_train, df_train['N_category']))
    
    for fold in config.train_params.selected_folds:
        print('start fold :', fold)
        config.start_time = time.strftime('%Y-%m-%d_%I:%M', time.localtime(time.time()))
        
        tt = df_train.loc[splits[fold][0]].reset_index(drop=True)#[:32]
        vv = df_train.loc[splits[fold][1]].reset_index(drop=True)#[:32]
        train_transforms, valid_transforms = train_get_transforms(config.train_params.img_size), valid_get_transforms()

        config.train_dataset = BC_Dataset(config, tt, img_size=config.train_params.img_size, transform=train_transforms)
        config.valid_dataset = BC_Dataset(config, vv, img_size=config.train_params.img_size, transform=valid_transforms)
        
        print(config.train_dataset[0]['img'].shape)
        print(config.train_dataset[0]['label'])
        
        lr_monitor = LearningRateMonitor(logging_interval='step') # ['epoch', 'step']
        checkpoints = ModelCheckpoint(config.output_dir + config.ver + '/', 
                                    #   monitor='total_val_loss', 
                                    monitor='total_val_f1_score',
                                    mode='max', 
                                    filename=f'5fold_{fold}__' + '{epoch}_{total_train_loss:.5f}_{total_train_f1_score:.5f}_{total_val_loss:.5f}_{total_val_f1_score:.5f}')
        model = pl_Wrapper(config)
        
        if config.gpu.mps:
            # tb_logger = pl_loggers.TensorBoardLogger(save_dir=f'{config.start_time}_{config.ver}_5fold_{fold}')
            trainer = pl.Trainer(
                                max_epochs=config.train_params.epochs, 
                                accelerator='mps', 
                                devices=1,
                                log_every_n_steps=30,
                                # gradient_clip_val=1000, gradient_clip_algorithm='value', # defalut : [norm, value]
                                amp_backend='native', precision=16, # amp_backend default : native
                                callbacks=[checkpoints, lr_monitor], 
                                # logger=tb_logger,
                                # limit_train_batches=10, 
                                # limit_val_batches=5
                                )
            
        else:
            from pytorch_lightning.loggers import WandbLogger
            logger = WandbLogger(name=f'{config.start_time}_{config.ver}_5fold_{fold}', 
                                    project='cancer', 
                                    config={key:config.__dict__[key] for key in config.__dict__.keys() if '__' not in key},
                                    )
            trainer = pl.Trainer(
                                max_epochs=config.train_params.epochs, 
                                accelerator='cuda', 
                                devices=1,
                                log_every_n_steps=30,
                                # gradient_clip_val=1000, gradient_clip_algorithm='value', # defalut : [norm, value]
                                amp_backend='native', precision=16, # amp_backend default : native
                                callbacks=[checkpoints, lr_monitor], 
                                logger=logger
                                )
                                
        trainer.fit(model)
        
        del model, trainer
        wandb.finish()


if __name__ == '__main__':

    ########## ENVIRONMENT SETUP ############
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument
    
    arg('--conf_path', type=str, default='base_config.yaml')

    # arg('--SEED', type=int, default=42)
    # arg('--DEVICE', type=str, default='cuda', choices=['cpu', 'cuda', 'mps'])

    # arg('--EPOCHS', type=int, default=50)#13
    # arg('--BATCH_SIZE', type=int, default=32)
    # arg('--VAL_BATCH_SIZE', type=int, default=64)

    # # optimizer
    # arg('--LR', type=float, default=1e-5) # cnn : 0.0025, rnn : [1e-5, 0.000025] xlm-roberta-large : 1e-5
    # arg('--WEIGHT_DECAY', type=float, default=1e-8) # cnn : 1e-3, rnn : 1e-8
    # arg('--sch_cycle', type=int, default=5)
    
    # # model
    # arg('--RNN_MODEL', type=str, default="lighthouse/mdeberta-v3-base-kor-further")
    # arg('--warmup_train', type=bool, default=False)
    
    # # save
    # arg('--OUTPUT_DIR', type=str, default='models/') 

    args = parser.parse_args()
    
    with open('conf/'+args.conf_path) as f:
        conf_yaml = yaml.safe_load(f)

    config = obj(conf_yaml)
    config.train_params.init_lr = float(config.train_params.init_lr)
    config.train_params.min_lr = float(config.train_params.min_lr)

    seed_everything(config.train_params.seed)
    main(config)
