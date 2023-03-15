import pandas as pd
import torch
from AntibodyFunctionPredictDatasets_forSNE import AntibodyFunctionPredictDatasets_forSNE,AntibodyFunctionPredictDatasets_forSNE_ProtTrans,AntibodyFunctionPredictDatasets_forHLSNE_CoV_AbDab
from Toolkit_forSNE import data_split_adj,OAS_paired_sampling
from torch.utils.data import DataLoader
import h5py
import numpy as np
from transformers import BertModel, BertTokenizer,T5EncoderModel, T5Tokenizer
import re
import os
import requests
from tqdm.auto import tqdm
import gc
from igfold import IgFoldRunner



def HL_seg_by_secondstructure(data_path,vocabs_path_H, vocabs_path_L,H_pretrain_model_path,L_pretrain_model_path,out_path,label_map_dic_lis):
    
    all_data_df = pd.read_csv(data_path)
    sampled_data_dic = OAS_paired_sampling(all_data_df)
    for  key,value in sampled_data_dic.items():
        data = AntibodyFunctionPredictDatasets_forHLSNE_CoV_AbDab(value,vocabs_path_H,vocabs_path_L,key,label_map_dic_lis)
        dataloader = DataLoader(data,batch_size=512)

       
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_H_antibert_sencondfstructure = BertModel.from_pretrained(H_pretrain_model_path,output_hidden_states=True, return_dict=True,add_pooling_layer = False)
        model_L_antibert_sencondfstructure = BertModel.from_pretrained(L_pretrain_model_path,output_hidden_states=True, return_dict=True,add_pooling_layer = False)
        

        model_H_antibert_sencondfstructure.to(device)

        model_H_antibert_sencondfstructure.eval()

        model_L_antibert_sencondfstructure.to(device)

        model_L_antibert_sencondfstructure.eval()
        
        
        embedding_results = []
        label = []
        for i,data in enumerate(dataloader):
            with torch.no_grad():
                print(i)
                print(data['label'])
                
                data = {key: value.cuda() for key, value in data.items()}
                model_out_H = model_H_antibert_sencondfstructure(input_ids = data['H_ids'],attention_mask = data['H_attention_mask'])[0]
                model_out_L = model_L_antibert_sencondfstructure(input_ids = data['L_ids'],attention_mask = data['L_attention_mask'])[0]
                model_out_H = model_out_H.cpu().numpy()
                model_out_L = model_out_L.cpu().numpy()    
                for seq_num in range(len(model_out_H)):
                    seq_len_H = (data['H_attention_mask'][seq_num] == 1).sum()
                    seq_emd_H = model_out_H[seq_num][1:seq_len_H-1]
                    seq_emd_H = np.mean(seq_emd_H,axis=0,keepdims=False)

                    seq_len_L = (data['L_attention_mask'][seq_num] == 1).sum()
                    seq_emd_L = model_out_L[seq_num][1:seq_len_L-1]
                    seq_emd_L = np.mean(seq_emd_L,axis=0,keepdims=False)
                    
                    seq_emd = np.concatenate([seq_emd_H,seq_emd_L])
                    
                    embedding_results.append(seq_emd)
                
                label.append(data['label'].cpu().numpy())
            

def HL_ProtBert(data_path,out_path,label_map_dic_lis):
    
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

    all_data_df = pd.read_csv(data_path)
    sampled_data_dic = OAS_paired_sampling(all_data_df)
    for  key,value in sampled_data_dic.items():
        data = [{'H':' '.join(list(value['sequence_alignment_aa_heavy'].iloc[i].replace(" ", ""))),'L':' '.join(list(value['sequence_alignment_aa_light'].iloc[i].replace(" ", ""))),'label':label_map_dic_lis[key][value[key].iloc[i]]} for i in range(len(value))]
        dataset = AntibodyFunctionPredictDatasets_forSNE_ProtTrans(data) 
        dataloader = DataLoader(dataset,batch_size=512)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = BertModel.from_pretrained("Rostlab/prot_bert")

        model = model.to(device)

        model = model.eval()
        
        embedding_results = []
        label = []
        for i,data in enumerate(dataloader):
            ids_H = tokenizer.batch_encode_plus(data['H'], add_special_tokens=True, pad_to_max_length=True)
            ids_L = tokenizer.batch_encode_plus(data['L'], add_special_tokens=True, pad_to_max_length=True)
        
            with torch.no_grad():
                print(i)
                
                input_ids_H = torch.tensor(ids_H['input_ids']).to(device)
                attention_mask_H = torch.tensor(ids_H['attention_mask']).to(device)
                embedding_H =  model(input_ids=input_ids_H,attention_mask=attention_mask_H)[0]
                embedding_H = embedding_H.cpu().numpy()

                input_ids_L = torch.tensor(ids_L['input_ids']).to(device)
                attention_mask_L = torch.tensor(ids_L['attention_mask']).to(device)
                embedding_L =  model(input_ids=input_ids_L,attention_mask=attention_mask_L)[0]
                embedding_L = embedding_L.cpu().numpy()

                for seq_num in range(len(input_ids_H)):
                    seq_len_H = (attention_mask_H[seq_num] == 1).sum()
                    seq_emd_H = embedding_H[seq_num][1:seq_len_H-1]
                    seq_emd_H = np.mean(seq_emd_H,axis=0,keepdims=False)

                    seq_len_L = (attention_mask_L[seq_num] == 1).sum()
                    seq_emd_L = embedding_L[seq_num][1:seq_len_L-1]
                    seq_emd_L = np.mean(seq_emd_L,axis=0,keepdims=False)

                    seq_emd = np.concatenate([seq_emd_H,seq_emd_L])

                    embedding_results.append(seq_emd)
                
                label.append(data['label'].cpu().numpy())
            
        embedding_results = np.stack(embedding_results,axis=0)
        
        label = np.concatenate(label,axis=0)
        print(embedding_results.shape,len(label))
            
        
        out_path_i = out_path +'/' +'/OAS_pair_sequence_HL_ProtBert'+f'_{key}_'+'embedding.h5'
        b = h5py.File(out_path_i,"w")
        b.create_dataset('embedding',data = embedding_results)
        b.create_dataset('label',data=label,dtype=np.int)
        for fkey in b.keys():
            print(fkey)
            print(b[fkey].name)
            print(b[fkey].shape)
        b.close()

def HL_ProtT5_XL_UniRef50(data_path,out_path,label_map_dic_lis):
    
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )

    all_data_df = pd.read_csv(data_path)
    sampled_data_dic = OAS_paired_sampling(all_data_df)
    for  key,value in sampled_data_dic.items():
        data = [{'H':' '.join(list(value['sequence_alignment_aa_heavy'].iloc[i].replace(" ", ""))),'L':' '.join(list(value['sequence_alignment_aa_light'].iloc[i].replace(" ", ""))),'label':label_map_dic_lis[key][value[key].iloc[i]]} for i in range(len(value))]
        dataset = AntibodyFunctionPredictDatasets_forSNE_ProtTrans(data) 
        dataloader = DataLoader(dataset,batch_size=128)
        # print(data,len(data))
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        
        gc.collect()

        model = model.to(device)

        model = model.eval()

        embedding_results = []
        label = []
        for i,data in enumerate(dataloader):
            ids_H = tokenizer.batch_encode_plus(data['H'], add_special_tokens=True, pad_to_max_length=True)
            ids_L = tokenizer.batch_encode_plus(data['L'], add_special_tokens=True, pad_to_max_length=True)

            with torch.no_grad():
                print(i)
                input_ids_H = torch.tensor(ids_H['input_ids']).to(device)
                attention_mask_H = torch.tensor(ids_H['attention_mask']).to(device)
                embedding_H =  model(input_ids=input_ids_H,attention_mask=attention_mask_H)[0]
                embedding_H = embedding_H.cpu().numpy()

                input_ids_L = torch.tensor(ids_L['input_ids']).to(device)
                attention_mask_L = torch.tensor(ids_L['attention_mask']).to(device)
                embedding_L =  model(input_ids=input_ids_L,attention_mask=attention_mask_L)[0]
                embedding_L = embedding_L.cpu().numpy()

                for seq_num in range(len(input_ids_H)):
                    seq_len_H = (attention_mask_H[seq_num] == 1).sum()
                    seq_emd_H = embedding_H[seq_num][1:seq_len_H-1]
                    seq_emd_H = np.mean(seq_emd_H,axis=0,keepdims=False)

                    seq_len_L = (attention_mask_L[seq_num] == 1).sum()
                    seq_emd_L = embedding_L[seq_num][1:seq_len_L-1]
                    seq_emd_L = np.mean(seq_emd_L,axis=0,keepdims=False)

                    seq_emd = np.concatenate([seq_emd_H,seq_emd_L])

                    embedding_results.append(seq_emd)
                label.append(data['label'].cpu().numpy())
            
        embedding_results = np.stack(embedding_results,axis=0)
        label = np.concatenate(label,axis=0)
        print(embedding_results.shape,len(label))
            
        out_path_i = out_path +'/' +'/OAS_pair_sequence_HL_ProtT5_XL_UniRef50'+f'_{key}_'+'embedding.h5'
        b = h5py.File(out_path_i,"w")
        b.create_dataset('embedding',data = embedding_results)
        b.create_dataset('label',data=label,dtype=np.int)
        for fkey in b.keys():
            print(fkey)
            print(b[fkey].name)
            print(b[fkey].shape)
        b.close()
        print('得到表示h5文件')

def HL_AntiBERTy(data_path,out_path,label_map_dic_lis):
    all_data_df = pd.read_csv(data_path)
    sampled_data_dic = OAS_paired_sampling(all_data_df)
    for  key,value in sampled_data_dic.items():
        data = [{'H':value['sequence_alignment_aa_heavy'].iloc[i].replace(" ", ""),'L':value['sequence_alignment_aa_light'].iloc[i].replace(" ", ""),'label':label_map_dic_lis[key][value[key].iloc[i]]} for i in range(len(value))]
        dataset = AntibodyFunctionPredictDatasets_forSNE_ProtTrans(data) 
        dataloader = DataLoader(dataset,batch_size=1)
        
        embedding_results = []
        label = []
        for i,data in enumerate(dataloader) :
            print(i)
            sequences = {'H':data['H'],'L':data['L']}
            print(len(data['H'][0].replace(" ", ""))+len(data['L'][0].replace(" ", "")))

            igfold = IgFoldRunner()
            emb = igfold.embed(
            sequences=sequences, # Antibody sequences
            )
            embedding = emb.bert_embs.detach().cpu().numpy()
            seq_emd = np.mean(embedding,axis=1,keepdims=False)
            embedding_results.append(seq_emd)
    
            label.append(data['label'].cpu().numpy())
            

        embedding_results = np.concatenate(embedding_results,axis=0)
        
        label = np.concatenate(label,axis=0)
        print(embedding_results.shape,len(label))
            
        
        out_path_i = out_path +'/' +'/OAS_pair_sequence_HL_AntiBERTy'+f'_{key}_'+'embedding.h5'
        b = h5py.File(out_path_i,"w")
        b.create_dataset('embedding',data = embedding_results)
        b.create_dataset('label',data=label,dtype=np.int)
        for fkey in b.keys():
            print(fkey)
            print(b[fkey].name)
            print(b[fkey].shape)
        b.close()



if __name__ == '__main__':

    Species_label_map_dic = {'human':0,'mouse_BALB/c':1,'mouse_C57BL/6':2,'rat_SD':3} 
    BSource_label_map_dic = {'Lymph':0,'PBMC':1,'Tonsillectomy':2}
    BType_label_map_dic = {'ASC':0,'Memory-B-Cells':1,'Naive-B-Cells':2,'Plasmablast':3,'RV+B-Cells':4,'Unsorted-B-Cells':5}
    Disease_label_map_dic = {'HIV':0,'None':1,'Obstructive-Sleep-Apnea':2,'SARS-COV-2':3,'Tonsillitis':4,'Tonsillitis/Obstructive-Sleep-Apnea':5}
    label_map_dic_lis = {'Species':Species_label_map_dic,'BSource':BSource_label_map_dic,'BType':BType_label_map_dic,'Disease':Disease_label_map_dic}
    
   

    HL_seg_by_secondstructure(data_path='/home/luoxw/antibody_doctor_project/pretrain_bert/t-SNE/data/OAS_pair_sequence.csv',
                             vocabs_path_H='/home/luoxw/antibody_doctor_project/pretrain_bert/vocabs/seg_by_secondstructure_H_wordpiece/vocab.txt',
                             vocabs_path_L='/home/luoxw/antibody_doctor_project/pretrain_bert/vocabs/seg_by_secondstructure_L_wordpiece/vocab.txt',
                             H_pretrain_model_path='/home/luoxw/antibody_doctor_project/pretrain_bert/models/seg_by_secondstructure_all-X_H_first_base_h5_True_BS256_EPS10_LR5e-05_WD0.01_WS100000_num2/model/AntibodyPretrainedModel_ep9',
                             L_pretrain_model_path='/home/luoxw/antibody_doctor_project/pretrain_bert/models/seg_by_secondstructure_all-X_L_first_base_h5_True_BS256_EPS30_LR5e-05_WD0.01_WS100000_num2/model/AntibodyPretrainedModel_ep29',
                             out_path='/home/luoxw/antibody_doctor_project/pretrain_bert/t-SNE/data' ,
                             label_map_dic_lis=label_map_dic_lis)

    HL_ProtBert(data_path='/home/luoxw/antibody_doctor_project/pretrain_bert/t-SNE/data/OAS_pair_sequence.csv',
               out_path='/home/luoxw/antibody_doctor_project/pretrain_bert/t-SNE/data',
               label_map_dic_lis=label_map_dic_lis)

    HL_ProtT5_XL_UniRef50(data_path='/home/luoxw/antibody_doctor_project/pretrain_bert/t-SNE/data/OAS_pair_sequence.csv',
                         out_path='/home/luoxw/antibody_doctor_project/pretrain_bert/t-SNE/data',
                         label_map_dic_lis=label_map_dic_lis )

    HL_AntiBERTy(data_path='/home/luoxw/antibody_doctor_project/pretrain_bert/t-SNE/data/OAS_pair_sequence.csv',
                out_path='/home/luoxw/antibody_doctor_project/pretrain_bert/t-SNE/data',
                label_map_dic_lis=label_map_dic_lis  )

 
 