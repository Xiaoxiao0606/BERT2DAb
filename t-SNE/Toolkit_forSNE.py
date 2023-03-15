import numpy as np
import pandas as pd
from torchmetrics import F1
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import torch
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import KFold,train_test_split
from datetime import datetime
import numpy as np



def data_split_adj(csv_allseqs, fraction):
    """
    Create a collection of the data set and split into the
    training set and two test sets. Data set is adjusted to
    match the specified class split fraction, which determines
    the fraction of Ag+ sequences.

    Parameters
    ---
    Ag_pos: Dataframe of the Ag+ data set
    Ag_neg: Dataframe of the Ag- data set
    fraction: The desired fraction of Ag+ in the data set
    """

    
    Ag_pos = csv_allseqs[csv_allseqs['label'] == 1]
    Ag_neg = csv_allseqs[csv_allseqs['label'] == 0]

    # Calculate data sizes based on ratio
    data_size_pos = len(Ag_pos)/fraction
    data_size_neg = len(Ag_neg)/(1-fraction)

    # Adjust the length of the data frames to meet the ratio requirement
    if len(Ag_pos) <= len(Ag_neg):
        if data_size_neg < data_size_pos:
            Ag_pos1 = Ag_pos[0:int((data_size_neg*fraction))]
            Ag_neg1 = Ag_neg
            Unused = Ag_pos[int((data_size_neg*fraction)):len(Ag_pos)]

        if data_size_neg >= data_size_pos:
            Ag_pos1 = Ag_pos[0:int((data_size_pos*(fraction)))]
            Ag_neg1 = Ag_neg[0:int((data_size_pos*(1-fraction)))]
            Unused = pd.concat(
                [Ag_pos[int((data_size_pos*fraction)):len(Ag_pos)],
                 Ag_neg[int((data_size_pos*(1-fraction))):len(Ag_neg)]]
            )
    else:
        if data_size_pos < data_size_neg:
            Ag_pos1 = Ag_pos
            Ag_neg1 = Ag_neg[0:(int(data_size_pos*(1-fraction)))]
            Unused = Ag_pos[int((data_size_pos*fraction)):len(Ag_pos)]

        if data_size_pos >= data_size_neg:
            Ag_pos1 = Ag_pos[0:int((data_size_neg*(fraction)))]
            Ag_neg1 = Ag_neg[0:int((data_size_neg*(1-fraction)))]
            Unused = pd.concat(
                [Ag_pos[int((data_size_neg*fraction)):len(Ag_pos)],
                 Ag_neg[int((data_size_neg*(1-fraction))):len(Ag_neg)]]
            )

    # Combine the positive and negative data frames
    Ag_combined = pd.concat([Ag_pos1, Ag_neg1])
    Ag_combined = Ag_combined.drop_duplicates(subset='VH')
    Ag_combined = Ag_combined.sample(frac=1,random_state=19930606).reset_index(drop=True)

 

    return Ag_combined, Unused

def OAS_paired_sampling(csv_allseqs):
    data = csv_allseqs

    label_list = ['Species','BSource','BType','Disease']

    sampled_subset = {}

    for i in  label_list:
        _,sampled_set = train_test_split(data,test_size=0.05,stratify=data[i],random_state=19930606)
        sampled_subset[i] = sampled_set
    
    return sampled_subset