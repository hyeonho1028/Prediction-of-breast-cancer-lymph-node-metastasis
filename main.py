import os
import sys
import time
import wandb
import argparse

import pandas as pd
import numpy as np
# from tabulate import tabulate
from sklearn.model_selection import StratifiedKFold

import torch

# from src import AD_Dataset, Multi_AD_Dataset, train_get_transforms, valid_get_transforms
from src import pl_Wrapper

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src import seed_everything

# warnings
import warnings
warnings.filterwarnings('ignore')

# class config:
#     bs = args.BATCH_SIZE

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main(args):
    df_train = pd.read_csv('open/train.csv')
    df_test = pd.read_csv('open/test.csv')
    sub = pd.read_csv('open/sample_submission.csv')

    df_train['img_path'] = df_train['img_path'].apply(lambda x: x.replace('./', 'data/'))
    df_test['img_path'] = df_test['img_path'].apply(lambda x: x.replace('./', 'data/'))
    
    for fold in range(5):
        print('start fold :', fold)
        # args.start_time = time.strftime('%Y-%m-%d_%I:%M', time.localtime(time.time()))
        # logger = WandbLogger(name=f'{args.start_time}_{args.VER}_5fold_{fold}', 
        #                         project='AD_clf', 
        #                         config={key:args.__dict__[key] for key in args.__dict__.keys() if '__' not in key},
        #                         )
        
        # args.train_dataset = Multi_AD_Dataset(
        #                                 tt, 
        #                                 [tt['cat1'].map(label2idx_1).values,
        #                                 tt['cat2'].map(label2idx_2).values,
        #                                 tt['cat3'].map(label2idx).values],
        #                                 transform=train_transforms, tokenizer=tokenizer, mode='train')
        # args.valid_dataset = Multi_AD_Dataset(
        #                                 vv, 
        #                                 [vv['cat1'].map(label2idx_1).values,
        #                                 vv['cat2'].map(label2idx_2).values,
        #                                 vv['cat3'].map(label2idx).values],
        #                                 transform=valid_transforms, tokenizer=tokenizer, mode='valid')
        
        # # sample weight
        # target = tt['cat3'].map(label2idx).values
        # class_sample_count = np.array(
        #     [len(np.where(target == t)[0]) for t in np.unique(target)])
        # weight = 1. / class_sample_count
        # samples_weight = np.array([weight[t] for t in target])
        
        # args.samples_weight = torch.from_numpy(samples_weight)
        # args.samples_weight = args.samples_weight.double()

        # # print(args.train_dataset[0]['img'].shape)
        # print(args.train_dataset[0]['label'])
        
        # if args.DEVICE=='mps':
        #     from src.trainer import training
        #     training(args)
        # else:
        #     lr_monitor = LearningRateMonitor(logging_interval='step') # ['epoch', 'step']
        #     checkpoints = ModelCheckpoint(args.OUTPUT_DIR + args.VER + '/', 
        #                                 #   monitor='total_val_loss', 
        #                                 monitor='total_val_f1_score',
        #                                 mode='max', 
        #                                 filename=f'5fold_{fold}__' + '{epoch}_{total_train_loss:.5f}_{total_train_f1_score:.5f}_{total_val_loss:.5f}_{total_val_f1_score:.5f}')
                
        #     model = PL_AD(args)
        #     trainer = pl.Trainer(
        #                             max_epochs=args.EPOCHS, 
        #                             accelerator='gpu', 
        #                             devices=1,
        #                             log_every_n_steps=30,
        #                             # gradient_clip_val=1000, gradient_clip_algorithm='value', # defalut : [norm, value]
        #                             amp_backend='native', precision=16, # amp_backend default : native
        #                             callbacks=[checkpoints, lr_monitor], 
        #                             logger=logger
        #                             ) 
        #     trainer.fit(model)
        #     del model, trainer, 
        #     wandb.finish()

        # break

    

    

if __name__ == '__main__':

    ########## ENVIRONMENT SETUP ############
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument
    
    # arg('--VER', type=str, default='baseline')


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

    seed_everything(args.SEED)
    main(args)