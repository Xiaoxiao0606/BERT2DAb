from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

plt.rcParams['axes.unicode_minus']=False 

z_score_scaler = lambda x:(x-np.mean(x))/(np.std(x))


def Consistency_of_ordering_hot_figures():
    #Consistency_of_ordering
    fig,ax =  plt.subplots(4,2,**{'dpi':1024},figsize=(30, 15))
    plt.subplots_adjust(wspace=0.1)
    data_h05=pd.read_csv('./comtact_maps/data/logist_dataset/H_Consistency_of_pop_0.05.csv',header=None)
    data_l05=pd.read_csv('./comtact_maps/data/logist_dataset/L_Consistency_of_pop_0.05.csv',header=None)
    data_h10=pd.read_csv('./comtact_maps/data/logist_dataset/H_Consistency_of_pop_0.1.csv',header=None)
    data_l10=pd.read_csv('./comtact_maps/data/logist_dataset/L_Consistency_of_pop_0.1.csv',header=None)
    data_h20=pd.read_csv('./comtact_maps/data/logist_dataset/H_Consistency_of_pop_0.2.csv',header=None)
    data_l20=pd.read_csv('./comtact_maps/data/logist_dataset/L_Consistency_of_pop_0.2.csv',header=None)
    data_h30=pd.read_csv('./comtact_maps/data/logist_dataset/H_Consistency_of_pop_0.3.csv',header=None)
    data_l30=pd.read_csv('./comtact_maps/data/logist_dataset/L_Consistency_of_pop_0.3.csv',header=None)
    
    ax[0,0] = sns.heatmap(data_h05,cmap="YlGnBu",annot=True,ax=ax[0,0],fmt=".2%",linewidths=2, linecolor='red',vmin=0, vmax=1.0)
    ax[0,0].set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12'])
    ax[0,0].set_yticklabels(['1','2','3','4','5','6','7','8','9','10','11','12'])
    ax[0,0].set_ylabel('t = 0.05',fontsize = 14, fontweight ='bold')
    ax[0,0].set_title('Heavy chian',fontsize = 14, fontweight ='bold')
    ax[0,1] = sns.heatmap(data_l05,cmap="YlGnBu",annot=True,ax=ax[0,1],fmt=".2%",linewidths=2, linecolor='red',vmin=0, vmax=1.0)
    ax[0,1].set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12'])
    ax[0,1].set_yticklabels(['1','2','3','4','5','6','7','8','9','10','11','12'])
    ax[0,1].set_title('Light chian',fontsize = 14, fontweight ='bold')
    ax[1,0] = sns.heatmap(data_h10,cmap="YlGnBu",annot=True,ax=ax[1,0],fmt=".2%",linewidths=2, linecolor='red',vmin=0, vmax=1.0)#
    ax[1,0].set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12'])
    ax[1,0].set_yticklabels(['1','2','3','4','5','6','7','8','9','10','11','12'])
    ax[1,0].set_ylabel('t = 0.10', fontsize = 14, fontweight ='bold')
    ax[1,1] = sns.heatmap(data_l10,cmap="YlGnBu",annot=True,ax=ax[1,1],fmt=".2%",linewidths=2, linecolor='red',vmin=0, vmax=1.0)#
    ax[1,1].set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12'])
    ax[1,1].set_yticklabels(['1','2','3','4','5','6','7','8','9','10','11','12'])
    ax[2,0] = sns.heatmap(data_h20,cmap="YlGnBu",annot=True,ax=ax[2,0],fmt=".2%",linewidths=2, linecolor='red',vmin=0, vmax=1.0)#
    ax[2,0].set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12'])
    ax[2,0].set_yticklabels(['1','2','3','4','5','6','7','8','9','10','11','12'])
    ax[2,0].set_ylabel('t = 0.20', fontsize = 14, fontweight ='bold')
    ax[2,1] = sns.heatmap(data_l20,cmap="YlGnBu",annot=True,ax=ax[2,1],fmt=".2%",linewidths=2, linecolor='red',vmin=0, vmax=1.0)#
    ax[2,1].set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12'])
    ax[2,1].set_yticklabels(['1','2','3','4','5','6','7','8','9','10','11','12'])
    ax[3,0] = sns.heatmap(data_h30,cmap="YlGnBu",annot=True,ax=ax[3,0],fmt=".2%",linewidths=2, linecolor='red',vmin=0, vmax=1.0)#
    ax[3,0].set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12'])
    ax[3,0].set_yticklabels(['1','2','3','4','5','6','7','8','9','10','11','12'])
    ax[3,0].set_ylabel('t = 0.30', fontsize = 14, fontweight ='bold')
    ax[3,1] = sns.heatmap(data_l30,cmap="YlGnBu",annot=True,ax=ax[3,1],fmt=".2%",linewidths=2, linecolor='red',vmin=0, vmax=1.0)#
    ax[3,1].set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12'])
    ax[3,1].set_yticklabels(['1','2','3','4','5','6','7','8','9','10','11','12'])

    
    plt.tight_layout()
    plt.show()


    plt.savefig(f"./comtact_maps/Consistency_of_ordering.jpg",bbox_inches='tight',pad_inches=1)


if __name__ == '__main__':

    Consistency_of_ordering_hot_figures()
