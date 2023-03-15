from this import d
import torch
from transformers import BertTokenizer,BertModel
import os
import ast
import random

class AntibodyFunctionPredictDatasets_forSNE(torch.utils.data.Dataset):
    def __init__(self,data_df,vocabs_path_H,vocabs_path_L,train_mode=True, labeled=True):
        self.df = data_df
        self.train_mode = train_mode
        self.labeled = labeled
        self.Hchaintokenizer = BertTokenizer(vocabs_path_H,do_lower_case=False)
        self.Lchaintokenizer = BertTokenizer(vocabs_path_L,do_lower_case=False)
        self.max_len = 128
        

    def __getitem__(self, index):
        row = self.df.iloc[index]
        H_ids,H_attention_mask,L_ids,L_attention_mask= self.get_token_ids(row)
        if self.labeled:
            labels = self.get_label(row)
            return {'H_ids':H_ids, 'H_attention_mask':H_attention_mask, 'L_ids':L_ids,'L_attention_mask':L_attention_mask,'label':labels}
        else:
            return {'H_ids':H_ids, 'H_attention_mask':H_attention_mask, 'L_ids':L_ids,'L_attention_mask':L_attention_mask}


    def __len__(self):
        return len(self.df)

    

    def trim_input(self, Hchain, Lchain):
        Hchain_token = self.Hchaintokenizer.tokenize(Hchain)
        Lchain_token = self.Lchaintokenizer.tokenize(Lchain)

        return Hchain_token, Lchain_token

    def get_token_ids(self, row):
        H_tokens, L_tokens = self.trim_input(row['VH'], row['VK'])
        
        H_tokens = ['[CLS]'] + H_tokens + ['[SEP]'] 
        L_tokens = ['[CLS]'] + L_tokens + ['[SEP]']
        # print(H_tokens, L_tokens)

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
        return torch.tensor(row['label'].astype(int))

    def get_epitope_feature(self,row):
        if self.args.epitope_fearture  == 'no':
            return 999
        elif self.args.epitope_fearture  == 'PyProtein':
            lis = []
            dic = ast.literal_eval(row['epitope_PyProtein'])
            
            for i in dic.values():
                lis.append(i)
            return torch.tensor(lis)

    def input_region(self, row):
        Hchain_token = self.Hchaintokenizer.tokenize(row['VH'])
        Lchain_token = self.Lchaintokenizer.tokenize(row['VK'])
       
        H_region_index = ast.literal_eval(row['H_region_index_abYsisIMGT'])
        L_region_index = ast.literal_eval(row['L_region_index_abYsisIMGT'])
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


class AntibodyFunctionPredictDatasets_forSNE_ProtTrans(torch.utils.data.Dataset):
    def __init__(self,data):
        self.data_dict = data
    def __getitem__(self, index):
        dict = self.data_dict[index]
     
        return {'H':dict['H'], 'L':dict['L'],'label':dict['label']}
        
    def __len__(self):
        return len(self.data_dict)

class AntibodyFunctionPredictDatasets_forHLSNE_CoV_AbDab(torch.utils.data.Dataset):
    def __init__(self,data_df,vocabs_path_H,vocabs_path_L,label,label_map_dic_lis):
        self.df = data_df
        self.Hchaintokenizer = BertTokenizer(vocabs_path_H,do_lower_case=False)
        self.Lchaintokenizer = BertTokenizer(vocabs_path_L,do_lower_case=False)
        self.label = label
        self.label_map_dic_lis =label_map_dic_lis
        self.max_len = 128
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        H_ids,H_attention_mask,L_ids,L_attention_mask= self.get_token_ids(row)
        labels = self.get_label(row)
        return {'H_ids':H_ids, 'H_attention_mask':H_attention_mask, 'L_ids':L_ids,'L_attention_mask':L_attention_mask,'label':labels}
      


    def __len__(self):
        return len(self.df)

    

    def trim_input(self, Hchain, Lchain):
        Hchain_token = self.Hchaintokenizer.tokenize(Hchain)
        Lchain_token = self.Lchaintokenizer.tokenize(Lchain)

        return Hchain_token, Lchain_token

    def get_token_ids(self, row):
        H_tokens, L_tokens = self.trim_input(row['sequence_alignment_aa_heavy_secondstructure'], row['sequence_alignment_aa_light_secondstructure'])
        
        H_tokens = ['[CLS]'] + H_tokens + ['[SEP]'] 
        L_tokens = ['[CLS]'] + L_tokens + ['[SEP]']
        # print(H_tokens, L_tokens)

        H_token_ids = self.Hchaintokenizer.convert_tokens_to_ids(H_tokens)
        H_attention_mask = torch.tensor([1]*len(H_token_ids)+[0] * (self.max_len - len(H_token_ids)),dtype=torch.int64)

        L_token_ids = self.Lchaintokenizer.convert_tokens_to_ids(L_tokens)
        L_attention_mask = torch.tensor([1]*len(L_token_ids)+[0] * (self.max_len - len(L_token_ids)),dtype=torch.int64)
        
        #input_ids
        if len(H_token_ids) < self.max_len:
            H_token_ids += [3] * (self.max_len - len(H_token_ids))
        H_ids = torch.tensor(H_token_ids,dtype=torch.int64)

        
        if len(L_token_ids) < self.max_len:
            L_token_ids += [3] * (self.max_len - len(L_token_ids))
        L_ids = torch.tensor(L_token_ids,dtype=torch.int64)
        
        
    
        return H_ids,H_attention_mask,L_ids,L_attention_mask

    def get_label(self, row):
        return torch.tensor(self.label_map_dic_lis[self.label][row[self.label]],dtype=torch.int64)

    