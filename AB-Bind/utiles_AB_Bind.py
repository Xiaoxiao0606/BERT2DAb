import numpy as np
import pandas as pd
from torchmetrics import F1
from AntibodyFunctionPredictDatasets_AB_Bind import AntibodyFunctionPredictDatasets_forregression
from torch.utils.data import DataLoader
from AntibodyFunctionPredictModel_AB_Bind import AntibodyFunctionPredictModel_forregression
from torch.utils.tensorboard import SummaryWriter
import os
import torch
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from AntibodyFunctionPredicttrainer_AB_Bind import AntibodyFunctionPredicttrainer_forregression
from sklearn.model_selection import KFold
from datetime import datetime
from imblearn.over_sampling import RandomOverSampler
from http.client import LineTooLong
from concurrent.futures import process, thread
import os
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from platform import release
from re import A
import time
from tkinter import N
from aiohttp import request
import pandas as pd
import numpy
import pandas as pd
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import tensorflow as tf
import time
from functools import partial
from numpy import append
import threading
from sklearn.model_selection import train_test_split



def train_or_validation_forregression(args,
                    df_train,
                    df_val,
                    writer,
                    fold,
                    device,
                    best_RMSE):


    

    torch.cuda.empty_cache()
    
    
    model = AntibodyFunctionPredictModel_forregression(args,
 
                                        )
    model.to(device)

    model.train()

    # print(model)


    num_gpus = torch.cuda.device_count()

    
    r=[True if args.bert_post in ['alltoken_ConV2D_fc','alltoken_transformer_fc','alltoken_ConV1D_transformer_fc','alltoken_ConV&transformer_cat_fc','alltoken_transformer_meanpool_fc'] else False][0]
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank ],
                                                        output_device=args.local_rank,find_unused_parameters=r)
    
    AntibodyFunctionPredictDatasets_train = AntibodyFunctionPredictDatasets_forregression(args,
                                                                                          data_df=df_train,

                                                                            )

                                                                                
    train_sampler = DistributedSampler(AntibodyFunctionPredictDatasets_train)       

    train_loader = DataLoader(AntibodyFunctionPredictDatasets_train,batch_size= args.batch_size,sampler = train_sampler)


    total_steps = (len(AntibodyFunctionPredictDatasets_train) // (args.batch_size * torch.cuda.device_count())) * args.num_epoch_train if len(AntibodyFunctionPredictDatasets_train) % (args.batch_size * torch.cuda.device_count()) == 0 else (len(AntibodyFunctionPredictDatasets_train) // (args.batch_size * torch.cuda.device_count()) + 1) * args.num_epoch_train
    warmup_steps = int(total_steps * args.warmup_steps_ratio)
    
  
    trainer = AntibodyFunctionPredicttrainer_forregression(args,
                                             model=model,
                                             writer = writer,
                                             total_steps = total_steps,
                                             warmup_steps=warmup_steps,
                                             fold =fold,
                                         
                                            )

   
    for epoch in range(args.num_epoch_train):
        train_sampler.set_epoch(epoch)
        if args.local_rank == 0:  
            print('——'*10, f'第{fold}折 train Epoch {epoch + 1}/{args.num_epoch_train}', '——'*10)
        trainer.train(epoch=epoch,train_data=train_loader)
        torch.distributed.barrier()

    
  
    torch.distributed.barrier()

    
    if args.do_validation == True:

        
        AntibodyFunctionPredictDatasets_val = AntibodyFunctionPredictDatasets_forregression(args,data_df=df_val,
                                                                                           
                                                                            
                                                                                  )
        
        val_sampler = DistributedSampler(AntibodyFunctionPredictDatasets_val)

        val_loader = DataLoader(AntibodyFunctionPredictDatasets_val,batch_size = 2, shuffle=False,sampler =val_sampler)

        for epoch in range(args.num_epoch_validation):
            val_sampler.set_epoch(epoch)
            if args.local_rank == 0: 
                print('——'*10, f'{fold} validation Epoch {epoch + 1}/{args.num_epoch_validation}', '——'*10)
                sample_RMSE_collect_validation=[]
                sample_Pearson_collect_validation=[]
                sample_P_collect_validation=[]

            epoch_result=trainer.validation(epoch=epoch,val_data=val_loader)

            if args.local_rank == 0: 
                sample_RMSE_collect_validation.append(epoch_result['sample_RMSE_collect_validation'])
                sample_Pearson_collect_validation.append(epoch_result['sample_Pearson_collect_validation'])
                sample_P_collect_validation.append(epoch_result['sample_P_collect_validation'])
                
                
            torch.distributed.barrier()

        if args.local_rank == 0: 
            res = {'fold':fold,
                    'RMSE':np.mean(sample_RMSE_collect_validation),
                    'Pearson':np.mean(sample_Pearson_collect_validation),
                    'P_value':np.mean(sample_P_collect_validation),
                    }
            
            if res['RMSE'] < best_RMSE:
                os.chdir(os.path.join(args.finetuning_model_paths,args.fineturned_modelname,args.num_parameter_current,'log'))
                os.system('rm -r *')
                trainer.save(epoch=args.num_epoch_train,file_path=os.path.join(args.finetuning_model_paths,args.fineturned_modelname,args.num_parameter_current,'log',
                         datetime.strftime(datetime.now(),'%Y-%m-%d_%H:%M:%S')+f'_num_hyperparameter_set_{str(args.num_parameter_current)}_fold{fold}'
                        ))
       
        torch.distributed.barrier()
        
        if args.local_rank == 0:
            return res
        else:
            return
    
    else:
        if args.local_rank == 0: 
            os.chdir(os.path.join(args.finetuning_model_paths,args.fineturned_modelname,args.num_parameter_current,'log'))
            os.system('rm -r *')
            trainer.save(epoch=args.num_epoch_train,file_path=os.path.join(args.finetuning_model_paths,args.fineturned_modelname,args.num_parameter_current,'log',
                        datetime.strftime(datetime.now(),'%Y-%m-%d_%H:%M:%S')+f'_num_hyperparameter_set_{str(args.num_parameter_current)}'
                    ))



#====================================second structure============================================
SS_LIST = ["C", "H", "E", "T", "G", "S", "I", "B"]
ANGLE_NAMES_LIST = ["PHI", "PSI", "THETA", "TAU"]
FASTA_RESIDUE_LIST = ["A", "D", "N", "R", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
NB_RESIDUES = len(FASTA_RESIDUE_LIST)
RESIDUE_DICT = dict(zip(FASTA_RESIDUE_LIST, range(NB_RESIDUES)))

EXP_NAMES_LIST = ["ASA", "CN", "HSE_A_U", "HSE_A_D"]
EXP_MAXS_LIST = [330.0, 131.0, 76.0, 79.0]  # Maximums from our dataset
UPPER_LENGTH_LIMIT = 1024

def mul1(rows):
    idx,s=rows
    if "X" not in s[0]:
        return {'idx':str(idx),'s':s[0]}
    
def mul2(df):
    idx_space_list =[]
    seqs = ''
    csv_data = df
    for s in range(csv_data.shape[0] - 1):
        a = csv_data.iloc[s,1]
        b = csv_data.iloc[s+1,1]
        if a == b :
            continue
        else:
            idx_space_list.append(s)
    for a in range(csv_data.shape[0]):
        if a not in idx_space_list:
            seqs = seqs + csv_data.iloc[a,0]
        else:
            seqs = seqs + csv_data.iloc[a,0]
            seqs = seqs +' '
    return seqs

def read_fasta(fasta_files_dict_list):
    protein_names_list = []
    sequences_list = []
    for file in fasta_files_dict_list:
        if file != None :
            protein_names = []
            sequences = []
            name = file['idx']
            protein_names.append(name)
            sequences.append(file['s'])
            protein_names_list.append(protein_names)
            sequences_list.append(sequences)
    return protein_names_list, sequences_list

def fill_array_with_value(array: np.array, length_limit: int, value):
    array_length = len(array)

    filler = value * np.ones((length_limit - array_length, array.shape[1]), array.dtype)
    filled_array = np.concatenate((array, filler))

    return filled_array

def save_prediction_to_csv(resnames: str, pred_c: np.array):
    
    sequence_length = len(resnames)

    output_df = pd.DataFrame()
    output_df["resname"] = [r for r in resnames]

    def get_ss(one_hot):
        return [["C", "H", "E", "T", "G", "S", "I", "B"][idx] for idx in np.argmax(one_hot, axis=-1)]

    output_df["Q3"] = get_ss(pred_c[0][0])[:sequence_length]
    output_df["Q8"] = get_ss(pred_c[1][0])[:sequence_length]


    return output_df

def mul4(idx,residue_lists,pred_c):
    resnames = residue_lists[idx][0]
    pred_c = [pred_c[0][idx:idx+1], pred_c[1][idx:idx+1,:,:]]

    sequence_length = len(resnames)

    output_df = pd.DataFrame()
    output_df["resname"] = [r for r in resnames]

    def get_ss(one_hot):
        return [["C", "H", "E", "T", "G", "S", "I", "B"][idx] for idx in np.argmax(one_hot, axis=-1)]

    output_df["Q3"] = get_ss(pred_c[0][0])[:sequence_length]
    output_df["Q8"] = get_ss(pred_c[1][0])[:sequence_length]

    return output_df

def main(ensemble_c,fasta_files_dict_list):
    protein_names, residue_lists = read_fasta(fasta_files_dict_list)

    sequence_list =[]
    for resnames in residue_lists:


        sequence = to_categorical([RESIDUE_DICT[residue] for residue in resnames[0]], num_classes=NB_RESIDUES)
        sequence = fill_array_with_value(sequence, UPPER_LENGTH_LIMIT, 0)

        sequence_list.append(sequence)

    print(f"Generating prediction...")
    start = time.time()
    pred_c = ensemble_c.predict(np.array(sequence_list))
   
    return residue_lists,protein_names,pred_c
    

def seg_by_no_to_seg_by_secondstructure(seg_by_no_data_path,seg_by_secondstructure_data_path,models_folder,chunksize,GPU,test,is_multipleproccess):
    with tf.device(f'/device:{GPU}'):
        ensemble_c = load_model(os.path.join(models_folder, "unet_c_ensemble"))
        seg_by_no_data_chunk = pd.read_csv(seg_by_no_data_path,header=None,chunksize=chunksize)
        with open(seg_by_secondstructure_data_path,'w') as  c:
            for n,chunk in enumerate(seg_by_no_data_chunk) :
                sta = time.time()
                print(f'{n*len(chunk)}')
                

                #===============================================
                if is_multipleproccess == True:
                    pool1 = Pool()
                    res1 = pool1.map(mul1,chunk.iterrows())
                    pool1.close()
                    pool1.join()
                else :
                    res1 = []
                    for i in chunk.iterrows():
                        res1.append(mul1(i))
                
                
                residue_lists,protein_names,pred_c = main(ensemble_c = ensemble_c,fasta_files_dict_list = res1 )
                

                

                df_list = []
                for idx in [i for i in range(len(protein_names))]:
                    df_list.append(save_prediction_to_csv(residue_lists[idx][0],
                                                          [pred_c[0][idx:idx+1], pred_c[1][idx:idx+1,:,:]])) 
                
                
                #===============================================
                if is_multipleproccess == True:
                    
                    pool2 = Pool()
                    res2=pool2.map(mul2,df_list)
                    pool2.close()
                    pool2.join()
                else:
                    
                    res2 = []
                    for i in df_list:
                        res2.append(mul2(i))

                for d in res2:
                    c.write(d+'\n')
                print('one chuank time:',time.time()-sta)
            
                if test ==  True:
                   break
