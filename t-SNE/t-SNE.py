from xml.sax import handler
from matplotlib.offsetbox import bbox_artist
import numpy as np
from sklearn.manifold import TSNE
import h5py
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

def get_SNE_embedding():
    for i,model_name_ in enumerate(model_name):
        for l,label_name_ in enumerate(label_name):
            data = h5py.File(data_path +'/'+'OAS_pair_sequence_HL_'+model_name_+'_'+label_name_+'_embedding.h5','r')
            embedding = data['embedding']
            label = data['label']

            
            pca = PCA(n_components=50,random_state=19930606)
            embedding_pca = pca.fit_transform(embedding)

            
            tsne = TSNE(n_components=2,perplexity=50, learning_rate='auto', init='pca',n_iter=5000,method="barnes_hut",verbose=1,random_state=19930606)
            embedding_sne = tsne.fit_transform(embedding_pca)

            
            scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
            result = scaler.fit_transform(embedding_sne)

            b = h5py.File(data_path +'/'+'OAS_pair_sequence_HL_'+model_name_+'_'+label_name_+'_SNE_embedding_1018.h5',"w")
            b.create_dataset('embedding',data = result)
            b.create_dataset('label',data=label,dtype=np.int)
            for fkey in b.keys():
                print(fkey)
                print(b[fkey].name)
                print(b[fkey].shape)
            b.close()

    

def Vis():
    
    plt.switch_backend('agg')
    _,axes = plt.subplots(4,2,**{'figsize':(10,8),'dpi':400})
    _.subplots_adjust(bottom=0.2)
    label_map_dict = {
                      'BSource':['Lymph','PBMC','Tonsillectomy'],
                      'Disease':['HIV','None','Obstructive-Sleep-Apnea','SARS-COV-2','Tonsillitis','Tonsillitis/Obstructive-Sleep-Apnea']}


    
    
    KL=[[0.49,0.50,0.51,0.50],
        [0.75,0.75,0.76,0.75],
        [0.83,0.83,0.84,0.85],
        [0.85,0.86,0.87,0.87]]
    KL=[[0.50,0.50],
        [0.75,0.75],
        [0.83,0.85],
        [0.86,0.87]]
    num = [['a','b','c','d'],['e','f','g','h']]

    font = {'family':'Helvetica','weight':'bold','size':8}
    for  x in range(4):
        for  y in range(2):
        
            data = h5py.File(data_path +'/'+'OAS_pair_sequence_HL_'+model_name[x]+'_'+label_name[y]+'_SNE_embedding.h5','r')
            embedding = data['embedding']
            label = data['label']
            
            if y == 0 :
                if model_name[x] == 'antibert_secondstructure':
                    axes[x,y].set_ylabel('BERT2DAb',font)
                else:
                    axes[x,y].set_ylabel(model_name[x],font)
            axes[x,y].set_xlabel(num[y][x] + '     KL='+str(KL[x][y]),font)
            axes[x,y].xaxis.set_ticks([])
            axes[x,y].yaxis.set_ticks([])
            
            
            axes[x,y].spines['bottom'].set_linewidth(2)
            axes[x,y].spines['left'].set_linewidth(2)
            axes[x,y].spines['top'].set_linewidth(2)
            axes[x,y].spines['right'].set_linewidth(2)

            
            scatter = axes[x,y].scatter(embedding[:,0], embedding[:,1], c=label,cmap = plt.cm.Spectral,s=5,alpha = 0.5)
            

            if x ==3:
                axes[x,y].legend(handles = scatter.legend_elements()[0],labels = label_map_dict[label_name[y]] ,title = label_name[y],
                                 loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=False, shadow=False,prop = font)
           

    plt.show
    plt.savefig("./t-SNE/OAS_pair_squence_subsets_HL_4models_4labels_tSNE.jpg",bbox_inches = 'tight',dpi=400)



if __name__ == '__main__':
    data_path = './t-SNE/data'
    
    model_name = ['antibert_secondstructure',
                  'AntiBERTy',
                  'ProtBert',
                  'ProtT5_XL_UniRef50'
                ]
    
    label_name = ['BSource','Disease']

    

    Vis()

 
 


