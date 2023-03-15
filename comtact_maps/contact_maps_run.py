from Bio.PDB import PDBParser
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
import math as ma
from imblearn.over_sampling import SMOTE
np.set_printoptions(threshold=np.inf)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def get_token_contact_map(mab,seq,seq_second):

    p = PDBParser()
    s = p.get_structure(mab, f"./comtact_maps/data/TheraSAbDab_SeqStruc_pdb/{mab}_{seq}.pdb")                    

    for chains in s:
        for chain in chains:
            residue_CA_coordinate_col = np.zeros([len(chain),3]) 
            for i,residue in enumerate(chain):  
                if residue.get_resname() == 'GLY':           
                    for index,atom in enumerate(residue):
                        if atom.get_name() == 'CA':
                            residue_CA_atom_coordinate = atom.get_vector().get_array()
                            residue_CA_coordinate_col[i,:] = residue_CA_atom_coordinate
                else:
                    for index,atom in enumerate(residue):
                        if atom.get_name() == 'CB':
                            residue_CA_atom_coordinate = atom.get_vector().get_array()
                            residue_CA_coordinate_col[i,:] = residue_CA_atom_coordinate
    
    residue_CA_distance = np.zeros([len(chain),len(chain)])
    for x in range(len(chain)):
        for y in range(len(chain)):
            residue_CA_distance[x,y] = np.linalg.norm(residue_CA_coordinate_col[x]-residue_CA_coordinate_col[y])
    residue_CA_distance_8A = np.where(residue_CA_distance < 8,1,0)
   
    
    if seq == 'H':
        Hchaintokenizer = BertTokenizer('./vocabs/H_wordpiece/vocab.txt',do_lower_case=False)
        token_list = Hchaintokenizer.tokenize(seq_second)
        token_list = [t.replace('#','') for t in token_list ]
        print(token_list)
    elif seq == 'L':
        Lchaintokenizer = BertTokenizer('./vocabs/L_wordpiece/vocab.txt',do_lower_case=False)
        token_list = Lchaintokenizer.tokenize(seq_second)
        token_list = [t.replace('#','') for t in token_list ]
        print(token_list)

    lenth_t1 = 0
    lenth_t2 = 0

    token_CA_contact_pop = np.zeros([len(token_list),len(token_list)])
    for p in range(len(token_list)):
        lenth_t2 = 0
        for u in range(len(token_list)):
            t1 = token_list[p]
            len_t1 = len(t1)
            star_t1 = lenth_t1
            end_t1 = lenth_t1 + len_t1
            t2 = token_list[u]
            len_t2 = len(t2)
            star_t2 = lenth_t2
            end_t2 = lenth_t2 + len_t2
            token_CA_contact_pop[p,u] = np.sum(residue_CA_distance_8A[star_t1:end_t1,star_t2:end_t2])/(residue_CA_distance_8A[star_t1:end_t1,star_t2:end_t2].size)
            lenth_t2 = lenth_t2 + len_t2
        lenth_t1 = lenth_t1 + len_t1
    
    

  
    token_CA_contact_pop = np.where(token_CA_contact_pop > 0,1,0)
   
    return token_CA_contact_pop

def get_token_ids(Hchaintokenizer,Lchaintokenizer,row,seq):
    max_len = 128

    if seq == 'H':
        H_tokens = Hchaintokenizer.tokenize(row['Heavy Sequence second'])
        
        
        H_tokens = ['[CLS]'] + H_tokens + ['[SEP]'] 
       
        

        H_token_ids = Hchaintokenizer.convert_tokens_to_ids(H_tokens)
        H_attention_mask = torch.tensor([[1]*len(H_token_ids)+[0] * (max_len - len(H_token_ids))])

        
        
        
        if len(H_token_ids) < max_len:
            H_token_ids += [3] * (max_len - len(H_token_ids))
        H_ids = torch.tensor([H_token_ids])

        
       
       
        return {'H_ids':H_ids, 'H_attention_mask':H_attention_mask}

    elif seq == 'L':
        
        L_tokens = Lchaintokenizer.tokenize(row['Light Sequence second'])
        
        
        L_tokens = ['[CLS]'] + L_tokens + ['[SEP]']
        

        

        L_token_ids = Lchaintokenizer.convert_tokens_to_ids(L_tokens)
        L_attention_mask = torch.tensor([[1]*len(L_token_ids)+[0] * (max_len - len(L_token_ids))])
        
       
       

        
        if len(L_token_ids) < max_len:
            L_token_ids += [3] * (max_len - len(L_token_ids))
        L_ids = torch.tensor([L_token_ids])
        
       
        
        return {'L_ids':L_ids,'L_attention_mask':L_attention_mask}

def get_ascend_index(lst):
    res = []
    idx = -1
    length = len(lst)
    for i in range(length):
        idx = -1
        for j in range(length):
            if lst[j]<=lst[i]:
                idx +=1
        res.append(idx)
    return res

def token_contact_map_class(token_map):
    new =[]
    for x in range(token_map.shape[0]):
        indices = token_map[x][:]
        
        labels = ['0','1','2']     
        indices=pd.qcut(indices,3,labels=False)
        
        new.append(indices.tolist())
  
    return new

def softmax(x):
    
    sum_exp_a = np.sum(x)
    y = x / sum_exp_a
    return y

def get_logist_dataset(logit,pop):
    Hchainbert = BertModel.from_pretrained('w139700701/BERT2DAb_H',output_hidden_states=True, return_dict=True,add_pooling_layer = False,output_attentions=True)
    Lchainbert = BertModel.from_pretrained('w139700701/BERT2DAb_L',output_hidden_states=True, return_dict=True,add_pooling_layer = False,output_attentions=True)
    Hchaintokenizer = BertTokenizer('w139700701/BERT2DAb_H',do_lower_case=False)
    Lchaintokenizer = BertTokenizer('w139700701/BERT2DAb_L',do_lower_case=False)
    TheraSAbDab_SeqStruc_csv = pd.read_csv('./comtact_maps/data/TheraSAbDab_SeqStruc_OnlineDownload.csv')
    
    
    token_contact_map_H_list_CAcontact = []
    model_out_attention_H_list_softmax = []
    token_contact_map_L_list_CAcontact = []
    model_out_attention_L_list_softmax = []

    sc = StandardScaler()
   
    for i in range(len(TheraSAbDab_SeqStruc_csv)):
        if TheraSAbDab_SeqStruc_csv.iloc[i]['Therapeutic'] == 'Abagovomab':
            for seq in ['H','L']:
                if seq == 'H':
                    if TheraSAbDab_SeqStruc_csv.iloc[i]['Heavy Sequence'] != 'na':
                        #token contact map
                        token_contact_map = get_token_contact_map(mab = TheraSAbDab_SeqStruc_csv.iloc[i]['Therapeutic'],
                                            seq = seq,
                                            seq_second = TheraSAbDab_SeqStruc_csv.iloc[i]['Heavy Sequence second'])
                        print(token_contact_map)
                        token_contact_map_H_list_CAcontact.append(token_contact_map)

                        #attention
                        input_H=get_token_ids(Hchaintokenizer,Lchaintokenizer,TheraSAbDab_SeqStruc_csv.iloc[i],seq)
                        model_out_attention_H = Hchainbert(input_ids = torch.as_tensor(input_H['H_ids']),attention_mask =torch.as_tensor(input_H['H_attention_mask']))[-1]
                        new2_model_out_attention_H = []
                        for attention2 in model_out_attention_H:
                            new_att2 = np.zeros([1,12,token_contact_map.shape[0],token_contact_map.shape[0]])
                            for h in range(12):
                                attention_np2 = attention2.detach().numpy()[0]
                                att2 = attention_np2[h,1:token_contact_map.shape[0]+1,1:token_contact_map.shape[0]+1]
                               
                                for t in range(len(token_contact_map)):
                                   
                                    new_att2[0,h,t,:] = att2[t]
                                   
                            new2_model_out_attention_H.append(new_att2)
                        model_out_attention_H_list_softmax.append(new2_model_out_attention_H)
                       
                elif seq == 'L':
                    if TheraSAbDab_SeqStruc_csv.iloc[i]['Light Sequence'] != 'na':
                        
                        token_contact_map_L = get_token_contact_map(mab = TheraSAbDab_SeqStruc_csv.iloc[i]['Therapeutic'],
                                            seq = seq,
                                            seq_second = TheraSAbDab_SeqStruc_csv.iloc[i]['Light Sequence second'])
                        
                        token_contact_map_L_list_CAcontact.append(token_contact_map_L)

                        
                        input_L=get_token_ids(Hchaintokenizer,Lchaintokenizer,TheraSAbDab_SeqStruc_csv.iloc[i],seq)
                        model_out_attention_L = Lchainbert(input_ids = torch.as_tensor(input_L['L_ids']),attention_mask =torch.as_tensor(input_L['L_attention_mask']))[-1]
                        new2_model_out_attention_L = []
                        for attention2_L in model_out_attention_L:
                            new_att2_L = np.zeros([1,12,token_contact_map_L.shape[0],token_contact_map_L.shape[0]])
                            for h_L in range(12):
                                attention_np2_L = attention2_L.detach().numpy()[0]
                                att2_L = attention_np2_L[h_L,1:token_contact_map_L.shape[0]+1,1:token_contact_map_L.shape[0]+1]
                                
                                for t_L in range(len(token_contact_map_L)):
                                    
                                    new_att2_L[0,h_L,t_L,:] = att2_L[t_L]
                                   
                            new2_model_out_attention_L.append(new_att2_L)
                        model_out_attention_L_list_softmax.append(new2_model_out_attention_L)              
    
   

    if logit == True:
        for l in range(12):
            id_d = 0
            np_layer_logist_dataset = np.zeros([sum([i.size for i in token_contact_map_H_list_CAcontact]),14]) 
            for id,s in enumerate(model_out_attention_H_list_softmax):
                attention_l = s[l]
                for t_x in range(attention_l.shape[2]):
                    for t_y in range(attention_l.shape[3]):
                        n_logist = np.zeros(14)
                        n_logist[0] = token_contact_map_H_list_CAcontact[id][t_x][t_y]
                        n_logist[1] = str(id)
                        for h in range(12):
                            n_logist[h+2] = attention_l[:,h,t_x,t_y] 
                        np_layer_logist_dataset[id_d,:] =  n_logist
                        id_d += 1
           
            pd_data = pd.DataFrame(np_layer_logist_dataset,columns=['Distance_sortnum','Anti_id','head_0','head_1','head_2','head_3','head_4','head_5','head_6','head_7','head_8','head_9','head_10','head_11'])
            data_types_dict = {'Distance_sortnum': str,'Anti_id': str,'head_0': float,'head_1': float,'head_2': float,'head_3': float,'head_4': float,'head_5': float,'head_6': float,'head_7': float,'head_8': float,'head_9': float,'head_10': float,'head_11': float}
            pd_data = pd_data.astype(data_types_dict)
            pd_data.to_csv(f'./comtact_maps/data/logist_dataset/H_layer{str(l)}_attentionsoftmaxandtoken3c.csv')
           

        for l in range(12):
            id_d = 0
            np_layer_logist_dataset = np.zeros([sum([i.size for i in token_contact_map_L_list_CAcontact]),14]) 
            for id,s in enumerate(model_out_attention_L_list_softmax):
                attention_l = s[l]
                for t_x in range(attention_l.shape[2]):
                    for t_y in range(attention_l.shape[3]):
                        n_logist = np.zeros(14)
                        n_logist[0] = token_contact_map_L_list_CAcontact[id][t_x][t_y]
                        n_logist[1] = str(id)
                        for h in range(12):
                            n_logist[h+2] = attention_l[:,h,t_x,t_y]
                        np_layer_logist_dataset[id_d,:] =  n_logist
                        id_d += 1
            
            pd_data = pd.DataFrame(np_layer_logist_dataset,columns=['Distance_sortnum','Anti_id','head_0','head_1','head_2','head_3','head_4','head_5','head_6','head_7','head_8','head_9','head_10','head_11'])
            data_types_dict = {'Distance_sortnum': str,'Anti_id': str,'head_0': float,'head_1': float,'head_2': float,'head_3': float,'head_4': float,'head_5': float,'head_6': float,'head_7': float,'head_8': float,'head_9': float,'head_10': float,'head_11': float}
            pd_data = pd_data.astype(data_types_dict)
            pd_data.to_csv(f'./comtact_maps/data/logist_dataset/L_layer{str(l)}_attentionsoftmaxandtoken3c.csv')
            


    if pop == True:  
        for cut in [0.05,0.1,0.2,0.3]:
            np_Consistency_of_ordering_dataset = np.zeros([12,12],dtype=float)
            for l in range(12):
                for h in range(12):
                    Consist_list = []
                    for s in range(len(token_contact_map_H_list_CAcontact)):
                        map = token_contact_map_H_list_CAcontact[s]
                        attention = model_out_attention_H_list_softmax[s][l][:,h,:,:][0]
                        ind_atten = np.argwhere(attention > cut)
                        ind_map = np.argwhere(map == 1)
                        stac = np.vstack((ind_atten,ind_map))
                        unq,count = np.unique(stac,axis=0,return_counts=True)
                        Consist = unq[count>1].shape[0]/len(np.argwhere(attention > cut))
                        Consist_list.append(Consist)
                    np_Consistency_of_ordering_dataset[l,h] = np.mean(np.array(Consist_list))
            np.savetxt( f'./comtact_maps/data/logist_dataset/H_Consistency_of_pop_{str(cut)}.csv', np_Consistency_of_ordering_dataset, delimiter="," )

            np_Consistency_of_ordering_dataset = np.zeros([12,12],dtype=float)
            for l in range(12):
                for h in range(12):
                    Consist_list = []
                    for s in range(len(token_contact_map_L_list_CAcontact)):
                        map = token_contact_map_L_list_CAcontact[s]
                        attention = model_out_attention_L_list_softmax[s][l][:,h,:,:][0]
                        ind_atten = np.argwhere(attention > cut)
                        ind_map = np.argwhere(map == 1)
                        stac = np.vstack((ind_atten,ind_map))
                        unq,count = np.unique(stac,axis=0,return_counts=True)
                        Consist = unq[count>1].shape[0]/len(np.argwhere(attention > cut))
                        Consist_list.append(Consist)
                    np_Consistency_of_ordering_dataset[l,h] = np.mean(np.array(Consist_list))
            np.savetxt( f'./comtact_maps/data/logist_dataset/L_Consistency_of_pop_{str(cut)}.csv', np_Consistency_of_ordering_dataset, delimiter="," )

def sensitivityCalc(Predictions, Labels):
  
    tn, fp, fn, tp = confusion_matrix(Labels, Predictions).ravel()

    sensitivity = tp/(tp+fn)
    

    return sensitivity

def specificityCalc(Predictions, Labels):
    tn, fp, fn, tp = confusion_matrix(Labels, Predictions).ravel()

    specificity = tn/(tn+fp)
    

    return specificity
           
def run_logistic_model():
    smo = SMOTE(random_state = 19930606)
    for r in ['attentionsoftmaxandtoken3c']:
        for chain in ['H','L']:
            print(r,chain)
            metrics_csv = pd.DataFrame(columns=['Layer','accuracy','precision','recall','F1','Sensitivity','Specificity'])
            coef_csv = pd.DataFrame(columns=['Layer_heads','coef1','coef2','coef3','coef4','coef5','coef6','coef7','coef8','coef9','coef10','coef11','coef12'])
            for layer in range(12):
            
                data = pd.read_csv(f'./comtact_maps/data/logist_dataset/{chain}_layer{str(layer)}_{r}.csv')

            
                X = data.iloc[:,2:]
                y = data.iloc[:,1]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 19930606)
                X_train_smo, y_train_smo = smo.fit_resample(X_train, y_train) 

               
                
                lr = LogisticRegression(n_jobs = 10,max_iter=5000,solver='saga')
                

                lr.fit(X_train_smo, y_train_smo)


              
                metrics_csv = metrics_csv.append({'Layer':str(layer),
                                    'accuracy':lr.score(X_test, y_test),
                                    'precision':precision_score(y_true=y_test, y_pred=lr.predict(X_test)),
                                    'recall':recall_score(y_true=y_test, y_pred=lr.predict(X_test)),
                                    'F1':f1_score(y_true=y_test, y_pred=lr.predict(X_test)),
                                    'Sensitivity':sensitivityCalc(Predictions=lr.predict(X_test),Labels=y_test),
                                    'Specificity':specificityCalc(Predictions=lr.predict(X_test),Labels=y_test)},ignore_index = True)

                print('Test accuracy:', lr.score(X_test, y_test))
                print('Test precision:', precision_score(y_true=y_test, y_pred=lr.predict(X_test)))
                print('Test recall:', recall_score(y_true=y_test, y_pred=lr.predict(X_test)))
                print('Test F1:', f1_score(y_true=y_test, y_pred=lr.predict(X_test)))
                print('Sensitivity:', sensitivityCalc(Predictions=lr.predict(X_test),Labels=y_test))
                print('Specificity:', specificityCalc(Predictions=lr.predict(X_test),Labels=y_test))

                
                for m  in lr.coef_ :
                    coef_csv = coef_csv.append({'Layer_heads':str(layer),
                                                'coef1':m[1],
                                                'coef2':m[2],
                                                'coef3':m[3],
                                                'coef4':m[4],
                                                'coef5':m[5],
                                                'coef6':m[6],
                                                'coef7':m[7],
                                                'coef8':m[8],
                                                'coef9':m[9],
                                                'coef10':m[10],
                                                'coef11':m[11],
                                                'coef12':m[12]},ignore_index=True)
                

            metrics_csv.to_csv(f'./comtact_maps/data/results/logist_metrics_{chain}_{r}.csv')
            coef_csv.to_csv(f'./data/results/logist_coef_{chain}_{r}.csv')

if __name__=='__main__':
    
    
    get_logist_dataset(logit=True,pop=False)

    
    # run_logistic_model()
    





            


