from dataclasses import replace
import pandas as pd
import time
from utiles_AB_Bind import seg_by_no_to_seg_by_secondstructure

pd.set_option('display.max_rows', 1000) 
pd.set_option('display.max_columns', 1000) 

def get_seq(x,i):
    
    for seq_type,seq_index in x['Partners_index'].items():
        if  seq_type == 'H':
            H_mut =  x[f'Mutation_sequense_{str(seq_index)}']
            H_org =  x[f'Original_sequense_{str(seq_index)}']
        elif  seq_type == 'L':
            L_mut =  x[f'Mutation_sequense_{str(seq_index)}']
            L_org =  x[f'Original_sequense_{str(seq_index)}']
        else:
            epitope_mut =  x[f'Mutation_sequense_{str(seq_index)}']
            epitope_org =  x[f'Original_sequense_{str(seq_index)}']
    list = [H_mut,L_mut,epitope_mut,H_org,L_org,epitope_org]
    return list[i]
    
def seq_length(x,idx_dict,i):
    if x['#PDB'] == '3BDY':
        H_mut = x['H_mut'][1:idx_dict[x['#PDB']][0]]
        L_mut = x['L_mut'][0:idx_dict[x['#PDB']][1]]
        H_org = x['H_org'][1:idx_dict[x['#PDB']][0]]
        L_org = x['L_org'][0:idx_dict[x['#PDB']][1]]
    else:
        H_mut = x['H_mut'][0:idx_dict[x['#PDB']][0]]
        L_mut = x['L_mut'][0:idx_dict[x['#PDB']][1]]
        H_org = x['H_org'][0:idx_dict[x['#PDB']][0]]
        L_org = x['L_org'][0:idx_dict[x['#PDB']][1]]
    lis = [H_mut,L_mut,H_org,L_org]
    return lis[i]

def HL_VW(x,i):
    if x['Partners(A_B)'] == 'HL_VW':
        if i == 'Mutation_sequense_3':
            s = x['Mutation_sequense_3'] + x['Mutation_sequense_4']
            x['Mutation_sequense_4'] = ''
            return s
        elif i == 'Original_sequense_3':
            s = x['Original_sequense_3'] + x['Original_sequense_4']
            x['Mutation_sequense_4'] = ''
            return s
        
    else:
        if i == 'Mutation_sequense_3':
            return x['Mutation_sequense_3'] 
        elif i == 'Original_sequense_3':
            return x['Original_sequense_3'] 

data1 = pd.read_csv('./AB-Bind/data/AB-Bind_experimental_data_1096.csv',encoding='gbk')
data2 = pd.read_csv('./AB-Bind/data/AB-Bind_experimental_data_allMut_use.csv',encoding='gbk')


data2['Mutation'] = data2['Mutation'].str.replace(',', '')
print(len(data1),len(data2))

data_new = pd.merge(data2,data1,on=['#PDB','Mutation'],how='left')
print(len(data_new))


data_new['Partners(A_B)'][data_new['#PDB'] == '3NPS'] = 'A_HL'
for i in ['Mutation_sequense_3','Original_sequense_3']:
    data_new[i] = data_new.apply(HL_VW,axis=1,args=(i,))
data_new['Partners(A_B)'][data_new['Partners(A_B)'] == 'HL_VW'] = 'HL_c'
print(len(data_new))



idx = list(data_new[(data_new['Partners(A_B)'].str.contains('HL')) ].index)#| (data_new['#PDB'] == '3NPS')
data_new = data_new.loc[idx]
print(len(data_new))



data_new['Partners(A_B)_new'] = data_new['Partners(A_B)'].apply(lambda x:x.replace('_',''))
data_new['Partners_num'] = data_new['Partners(A_B)_new'].apply(lambda x:len(x))


idx = list(data_new[data_new['Partners_num'] == 3 ].index)
data_new = data_new.loc[idx]
print(len(data_new))

data_new['Partners_index'] = data_new['Partners(A_B)_new'].apply(lambda x:{item:idx+1 for idx,item in enumerate(x)})

for idx,s in enumerate([ 'H_mut','L_mut','epitope_mut','H_org','L_org','epitope_org']):
    data_new[s] = data_new.apply(get_seq,axis= 1,args=(idx,))

print(len(data_new))



seq_idx = {'1DQJ':(113,107),
           '1JRH':(122,106),
           '1MHP':(118,105),
           '1MLC':(116,107),
           '1N8Z':(120,107),
           '1VFB':(116,107),
           '1YY9':(119,107),
           '2JEL':(118,112),
           '2NY7':(127,108),
           '2NYY':(117,111),
           '2NZ9':(117,111),
           '3HFM':(113,107),
           '3NGB':(121,98),
           '3NPS':(125,106),
           '3BDY':(121,114),
           '3BE1':(120,114),
           '3BN9':(128,110),
           '1BJ1':(123,110),
           '1CZ8':(123,110),
           'HM_1YY9':(119,107),
           'HM_2NYY':(117,114),
           'HM_2NZ9':(117,114),
           'HM_3BN9':(128,110)}
for i,s in enumerate([ 'H_mut','L_mut','H_org','L_org']):
   data_new[s]  = data_new.apply(seq_length,axis= 1,args=(seq_idx,i))

data_new['H_mut'].to_csv('./AB-Bind/data/AB_Bind_postprocess_H_mut.txt',header=False,index=False)
data_new['L_mut'].to_csv('./AB-Bind/data/AB_Bind_postprocess_L_mut.txt',header=False,index=False)
data_new['H_org'].to_csv('./AB-Bind/data/AB_Bind_postprocess_H_org.txt',header=False,index=False)
data_new['L_org'].to_csv('./AB-Bind/data/AB_Bind_postprocess_L_org.txt',header=False,index=False)



for idx,i in enumerate(['./data/AB_Bind_postprocess_H_mut',
          './AB-Bind/data/AB_Bind_postprocess_L_mut',
          './AB-Bind/data/AB_Bind_postprocess_H_org',
          './AB-Bind/data/AB_Bind_postprocess_L_org']):
    seg_by_no_data_path = i +'.txt'
    seg_by_secondstructure_data_path = i + '_secondstructure.txt'
    models_folder = "./proteinUnet/data/models"
    star = time.time()
    seg_by_no_to_seg_by_secondstructure(seg_by_no_data_path=seg_by_no_data_path,
                                        seg_by_secondstructure_data_path=seg_by_secondstructure_data_path,
                                        models_folder=models_folder,
                                        chunksize = 5000,
                                        GPU='gpu:1',
                                        test = False,
                                        is_multipleproccess = False)
    end = time.time()
    print(end-star)
    
    with open(i + '_secondstructure.txt','r') as f:
        s = f.readlines()
        s = [i.splitlines()[0] for i in s]
    
    n = ['H_mut_secondstructure','L_mut_secondstructure','H_org_secondstructure','L_org_secondstructure']
    data_new[n[idx]] = s





data_new.to_csv('./AB-Bind/data/AB-Bind_experimental_data_allMut_addsequenceSecondstructure.csv')



