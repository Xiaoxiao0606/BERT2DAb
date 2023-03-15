from this import d
import torch
from transformers import BertTokenizer,BertModel
import os
import ast
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np

class AntibodyFunctionPredictDatasets(torch.utils.data.Dataset):
    def __init__(self, args,data_df,train_mode=True, labeled=True):
        self.df = data_df
        self.train_mode = train_mode
        self.labeled = labeled
        self.Hchaintokenizer = BertTokenizer('w139700701/BERT2DAb_H',do_lower_case=False)
        self.Lchaintokenizer = BertTokenizer('w139700701/BERT2DAb_L',do_lower_case=False)
       
        if args.seg == "seg_by_secondstructure":
            self.max_len = 128
        elif args.seg == "seg_by_no":
            self.max_len = 256
        self.args = args

        

    def __getitem__(self, index):
       
        row = self.df.iloc[index]
        H_ids,H_attention_mask,L_ids,L_attention_mask= self.get_token_ids(row)
        epitope_feature = self.get_epitope_feature(row)
        H_region,L_region = self.input_region(row)
        if self.labeled:
            labels = self.get_label(row)
            return {'H_ids':H_ids, 'H_attention_mask':H_attention_mask, 'L_ids':L_ids,'L_attention_mask':L_attention_mask,'label':labels,'epitope_feature': epitope_feature,'H_region':H_region,'L_region':L_region}
        else:
            return {'H_ids':H_ids, 'H_attention_mask':H_attention_mask, 'L_ids':L_ids,'L_attention_mask':L_attention_mask,'epitope_feature': epitope_feature,'H_region':H_region,'L_region':L_region}


    def __len__(self):
        return len(self.df)

    

    def trim_input(self, Hchain, Lchain):
        Hchain_token = self.Hchaintokenizer.tokenize(Hchain)
        Lchain_token = self.Lchaintokenizer.tokenize(Lchain)

        return Hchain_token, Lchain_token

    def get_token_ids(self, row):
        H_tokens, L_tokens = self.trim_input(row['VH_secondstructure'], row['VL_secondstructure'])
        
        H_tokens = ['[CLS]'] + H_tokens + ['[SEP]'] 
        L_tokens = ['[CLS]'] + L_tokens + ['[SEP]']
        

        H_token_ids = self.Hchaintokenizer.convert_tokens_to_ids(H_tokens)
        H_attention_mask = torch.tensor([1]*len(H_token_ids)+[0] * (self.max_len - len(H_token_ids)))

        L_token_ids = self.Lchaintokenizer.convert_tokens_to_ids(L_tokens)
        L_attention_mask = torch.tensor([1]*len(L_token_ids)+[0] * (self.max_len - len(L_token_ids)))
        
        #input_ids
        if len(H_token_ids) < self.max_len:
            H_token_ids += [3] * (self.max_len - len(H_token_ids))
        H_ids = torch.tensor(H_token_ids)

        
        if len(L_token_ids) < self.max_len:
            L_token_ids += [3] * (self.max_len - len(L_token_ids))
        L_ids = torch.tensor(L_token_ids)
        
        
        return H_ids,H_attention_mask,L_ids,L_attention_mask

    def get_label(self, row):
        return torch.tensor(row[self.args.Y_name].astype(int))

    def get_epitope_feature(self,row):
        if self.args.epitope_fearture  == 'no':
            return 999
        elif self.args.epitope_fearture  == 'PyProtein':
            lis = []
            dic = ast.literal_eval(row['Epitope_PyProtein'])
            for i in dic.values():
                lis.append(i)
            return torch.tensor(lis)

    def input_region(self, row):
        Hchain_token = self.Hchaintokenizer.tokenize(row['VH_secondstructure'])
        Lchain_token = self.Lchaintokenizer.tokenize(row['VL_secondstructure'])
        H_region_index = ast.literal_eval(row['H_region_index_AbRSAIMGT'])
        L_region_index = ast.literal_eval(row['L_region_index_AbRSAIMGT'])
        H_token_index = []
        L_token_index = []
        Hchain_token_index = []
        Lchain_token_index = []


        Hl=0
        Ll=0

        for i in Hchain_token:
            Hl += len(i.lstrip('#'))
            Hchain_token_index.append(Hl)
        for i in Lchain_token:
            Ll += len(i.lstrip('#'))
            Lchain_token_index.append(Ll)       

        idx_H = 0
        idx_L = 0
        
        for idx,i in  enumerate(Hchain_token_index):
           
            if i < H_region_index[idx_H]:
                H_token_index.append((idx_H,idx+1))
            elif i == H_region_index[idx_H]:
                H_token_index.append((idx_H,idx+1))
                idx_H += 1
            else:
                H_token_index.append((idx_H,idx+1))
                H_token_index.append((idx_H + 1,idx+1))

                
                idx_H += 1

        for idx,i in  enumerate(Lchain_token_index):
            if i <= L_region_index[idx_L]:
                L_token_index.append((idx_L,idx+1))
            elif i == L_region_index[idx_L]:
                L_token_index.append((idx_L,idx+1))
                idx_L += 1
            else:
                L_token_index.append((idx_L,idx+1))
                L_token_index.append((idx_L + 1,idx+1))
                
                idx_L += 1

        H_token_index += [(-1,-1)] * (100 - len(H_token_index))
        L_token_index += [(-1,-1)] * (100 - len(L_token_index))

        return torch.tensor(H_token_index,dtype=torch.int8),torch.tensor(L_token_index,dtype=torch.int8)
    

