from ast import arg
from cProfile import label
from tkinter import N
from tkinter.messagebox import NO
from datasets import load_dataset
import pandas as pd
from datasets import load_metric,list_metrics
from torch import frac, nn, relu
import torch
import pandas as pd
from tqdm import tqdm, trange
from datetime import datetime
from torchsummaryX import summary
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import argparse
from utiles_CoV_AbDab import data_sampling_for_CoV_AbDab,train_or_validation_forbinary
import contextlib
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import warnings



def parse():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-epval", "--num_epoch_validation", 
                        type=int, 
                        default=1,
                        help="num_epoch_validation")
    parser.add_argument("-d", "--data_path", 
                        type=str, 
                        default='./CoV-AbDab/data',
                        help="data_path")
    parser.add_argument("-v", "--vocabs_path", 
                        type=str, 
                        default='./vocabs',
                        help="vocabs_path")
    parser.add_argument("-f", "--finetuning_model_paths", 
                        type=str, 
                        default='./CoV-AbDab/models',
                        help="finetuning_model_paths")
    
    
    parser.add_argument("--do_validation", default=True, type=bool)
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--warmup_steps_ratio", type=float, default=0.1, help="warmup-steps")
    parser.add_argument("--log_freq", type=int, default=50, help="printing loss every n iter: setting n")


    return parser
    
def main(args=None,
         anti =None,
         N=None,
         freeze=None,
         batch_size=None,
         drop_out=None,
         seg=None,
         bert_post = None,
         epitope_fearture = None,
         layer =None,
         compound_mode =None,
         lr = None,
         adam_weight_decay = None,
         num_parameter_current =None,
         epoch = None,
         over_sample = None,
         dele = None):

    args.anti = anti
    args.N = N
    args.freeze = freeze
    args.batch_size = batch_size
    args.drop_out = drop_out
    args.seg = seg
    args.bert_post =bert_post
    args.epitope_fearture = epitope_fearture
    args.layer =layer
    args.compound_mode =compound_mode
    args.lr = lr
    args.adam_weight_decay = adam_weight_decay
    args.num_parameter_current = str(num_parameter_current)
    args.num_epoch_train = epoch
    
    args.over_sample = over_sample
    args.fineturned_modelname = anti + '_' + seg
    args.del_cls_pad = dele

    if args.seg == 'seg_by_secondstructure':
        args.vocabs_path_H = args.vocabs_path + '/H_wordpiece/vocab.txt'
        args.vocabs_path_L = args.vocabs_path + '/L_wordpiece/vocab.txt'
   

    
    if args.local_rank == 0: 
        os.makedirs(f'{args.finetuning_model_paths}/{args.fineturned_modelname}',exist_ok =True)
        os.makedirs(f'{args.finetuning_model_paths}/{args.fineturned_modelname}/'+args.num_parameter_current,exist_ok = True)
        os.makedirs(f'{args.finetuning_model_paths}/{args.fineturned_modelname}/'+args.num_parameter_current+'/log',exist_ok = True)
        os.makedirs(f'{args.finetuning_model_paths}/{args.fineturned_modelname}/'+args.num_parameter_current+'/runs',exist_ok = True)

    torch.cuda.set_device(args.local_rank )
    device=torch.device("cuda", args.local_rank )

    if num_parameter_current == args.num_parameter_current_start:
        if args.local_rank  != -1:        
            torch.distributed.init_process_group(backend="nccl", init_method='env://')
    
    writer = SummaryWriter(log_dir=os.path.join(args.finetuning_model_paths,args.fineturned_modelname,args.num_parameter_current,
                           'runs',datetime.strftime(datetime.now(),'%Y-%m-%d_%H:%M:%S') + 
                           f'_num_hyperparameter_set_{str(args.num_parameter_current)}'),
                           comment='AntibodyFunctionPredictModel')


    #====================================================================================
    all_data = pd.read_csv(os.path.join(args.data_path,args.dataset))
    all_data = all_data.sample(frac=args.N,random_state=19930606)
    y = np.squeeze(all_data[args.Y_name])
    x = all_data.drop([args.Y_name], axis=1)
    skf = StratifiedKFold(n_splits=args.fold,random_state=19930606,shuffle=True)
    dataset_dict = {}
    fold = 1
    for train_idx,test_idx in skf.split(x,y):
        X_train, X_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train[args.Y_name] = y_train
        X_test[args.Y_name] = y_test
        dataset_dict[f'fold{fold}'] = (X_train,X_test)
        fold += 1

    print(args)

    if args.local_rank == 0: 
        with open(f'{args.finetuning_model_paths}/{args.fineturned_modelname}/'+'hyperparameter_set.txt', 'a+') as f:
            with contextlib.redirect_stdout(f):
                print(f'{num_parameter_current}start_time：',datetime.strftime(datetime.now(),'%Y-%m-%d_%H:%M:%S'))
                print(args)

    #===============================================================================================
    res_df = pd.DataFrame(columns=['fold',
                                        'sample_acc_tm_collect_validation_mean',
                                        'sample_F1Score_tm_collect_validation_mean',
                                        'sample_precision_tm_collect_validation_mean',
                                        'sample_recall_tm_collect_validation_mean',
                                        'sample_acc_tm_collect_test_mean',
                                        'sample_F1Score_tm_collect_test_mean',
                                        'sample_precision_tm_collect_test_mean',
                                        'sample_recall_tm_collect_test_mean'])

    best_f1 = 0.0

    for fold,(train_dataset,validation_dataset) in dataset_dict.items():
        torch.cuda.empty_cache()

        if args.local_rank == 0:
            print('**'*10, fold, 'Training','ing....', '**'*10)

        if args.over_sample == True:

            train_dataset = data_sampling_for_CoV_AbDab(args,train_dataset)

        result=train_or_validation_forbinary(args=args,
                        df_train =train_dataset,
                        df_val = validation_dataset,
                        writer=writer,
                        fold = int(fold[-1]),
                        device =device,
                        best_F1 = best_f1)

        if args.do_validation == True:
            if args.local_rank == 0: 
                res_df = res_df.append(result,ignore_index=True)
                if result['sample_F1Score_tm_collect_validation_mean'] > best_f1:
                    best_f1 = result['sample_F1Score_tm_collect_validation_mean']
                    

    torch.distributed.barrier()

    if args.local_rank == 0: 
        res_df.loc['mean'] = res_df.apply(lambda x: x.mean())
        res_df.loc['std'] = res_df.apply(lambda x: x.std())
        res_df.loc['mean','fold'] = 'mean'
        res_df.loc['std','fold'] = 'std'
    
    res_df.loc['namespace','fold'] = str(args)
        
    res_df.to_csv(os.path.join(args.finetuning_model_paths,args.fineturned_modelname, f'num_hyperparameter_set_{str(args.num_parameter_current)}_validation_metrics.csv'))

    

if __name__ == '__main__':
   
    
    NUM_PARAMETER_CURRENT_START = 1
    DATASET_NAME = 'CoV-AbDab_260722_original_bind_postprocess_addPyProteinRegionindex_Dropduplicates.csv' 
    Y_NAME = 'IS_bind_two'
    Nb_or_Ab = 'Ab'
    FOLD = 10
    #=================================================================================================
    parser = parse()
    args = parser.parse_args()
    args.num_parameter_current_start = NUM_PARAMETER_CURRENT_START
   
    args.dataset = DATASET_NAME
    args.Y_name = Y_NAME
    args.fold = FOLD
    args.Nb_or_Ab = Nb_or_Ab
    num_parameter_current = args.num_parameter_current_start
    a = 'CoV-AbDab'

    #=================================================================================================
    for seg in ['seg_by_secondstructure']: 
        for epoch in [50]:
            for b in [16]:
                for  lr in [5e-5]:
                    for wd in [0.01]:
                        for sample in [True]:
                            for n in [1.0]:
                                for s in [True]:
                                    for d in [0.1]:
                                        for p_b in ['alltoken_cat_fc']:
                                            for epi in ['PyProtein']:
                                                for layer in [(12,)]:
                                                    for com in ['mean']:
                                                        for dele in ['yes']:
                                                            main(args =args,
                                                                anti=a,
                                                                N=n,
                                                                freeze=s,
                                                                batch_size = b,
                                                                drop_out = d,
                                                                seg=seg,
                                                                bert_post= p_b,
                                                                epitope_fearture = epi,
                                                                layer = layer,
                                                                compound_mode = com,
                                                                lr = lr,
                                                                adam_weight_decay = wd,
                                                                num_parameter_current =num_parameter_current,
                                                                epoch = epoch,
                                                                
                                                                over_sample = sample,
                                                                dele = dele)
                                                            num_parameter_current += 1

   







    # model_vison()
