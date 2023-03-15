from pickletools import uint8
from torch import frac, nn, relu
from transformers import BertModel
import torch
import numpy as np
from model_bert_pytorch.transformer import TransformerBlock

class ScheduledOptim():


    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps,lr):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = lr

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

class AntibodyFunctionPredictModel_forregression(nn.Module):

    def __init__(self,args,hidden_size = 768 ):
        super(AntibodyFunctionPredictModel_forregression,self).__init__()
        self.H_pretrain_model_path = args.pretrained_model_paths_H
        self.L_pretrain_model_path = args.pretrained_model_paths_L
        self.embed_size = hidden_size
        self.drop_out = args.drop_out
        self.Hchainbert_mut = BertModel.from_pretrained("w139700701/BERT2DAb_H",output_hidden_states=True, return_dict=True,add_pooling_layer = False)
        self.Lchainbert_mut = BertModel.from_pretrained("w139700701/BERT2DAb_L",output_hidden_states=True, return_dict=True,add_pooling_layer = False)
        self.Hchainbert_org = BertModel.from_pretrained("w139700701/BERT2DAb_H",output_hidden_states=True, return_dict=True,add_pooling_layer = False)
        self.Lchainbert_org = BertModel.from_pretrained("w139700701/BERT2DAb_L",output_hidden_states=True, return_dict=True,add_pooling_layer = False)
        self.bert_post = args.bert_post
        self.layer =args.layer
        self.compound_mode = args.compound_mode
        for param in self.Hchainbert_mut.parameters():
            param.requires_grad = args.freeze
        for param in self.Lchainbert_mut.parameters():
            param.requires_grad = args.freeze
        for param in self.Hchainbert_org.parameters():
            param.requires_grad = args.freeze
        for param in self.Lchainbert_org.parameters():
            param.requires_grad = args.freeze
        self.args = args

        self.ln_mut = nn.LayerNorm(self.embed_size*2 + 147)
        self.fc1_mut = nn.Linear(self.embed_size*2+ 147,512)
        self.dp1_mut = nn.Dropout(self.drop_out)
        self.fc2_mut = nn.Linear(512,128)
        self.dp2_mut = nn.Dropout(self.drop_out)

        self.ln_org = nn.LayerNorm(self.embed_size*2 + 147)
        self.fc1_org = nn.Linear(self.embed_size*2+ 147,512)
        self.dp1_org = nn.Dropout(self.drop_out)
        self.fc2_org = nn.Linear(512,128)
        self.dp2_org = nn.Dropout(self.drop_out)



    def forward(self,H_ids_mut,H_attention_mask_mut,L_ids_mut,L_attention_mask_mut,
                     H_ids_org,H_attention_mask_org,L_ids_org,L_attention_mask_org,epitope_feature_mut,epitope_feature_org,training=True):
        Hchainbert_outputs_mut = self.Hchainbert_mut(input_ids = H_ids_mut,attention_mask = H_attention_mask_mut)
        Hchainbert_hidden_state_mut = [Hchainbert_outputs_mut.hidden_states[i] for i in self.layer]
        if len(Hchainbert_hidden_state_mut) > 1 :
            Hchainbert_hidden_state_mut = torch.stack(Hchainbert_hidden_state_mut,dim=2)
            if self.compound_mode == 'sum':
                Hchainbert_hidden_state_mut = torch.sum(Hchainbert_hidden_state_mut,dim=2,keepdim=False)
            elif self.compound_mode == 'mean':
                Hchainbert_hidden_state_mut = torch.mean(Hchainbert_hidden_state_mut,dim=2,keepdim=False)
        else:
            Hchainbert_hidden_state_mut = Hchainbert_hidden_state_mut[0]
       

        Lchainbert_outputs_mut = self.Lchainbert_mut(input_ids = L_ids_mut,attention_mask = L_attention_mask_mut)
        Lchainbert_hidden_state_mut = [Lchainbert_outputs_mut.hidden_states[i] for i in self.layer]
        if len(Lchainbert_hidden_state_mut) > 1 :
            Lchainbert_hidden_state_mut = torch.stack(Lchainbert_hidden_state_mut,dim=2)
            if self.compound_mode == 'sum':
                Lchainbert_hidden_state_mut = torch.sum(Lchainbert_hidden_state_mut,dim=2,keepdim=False)
            elif self.compound_mode == 'mean':
                Lchainbert_hidden_state_mut = torch.mean(Lchainbert_hidden_state_mut,dim=2,keepdim=False)
        else:
            Lchainbert_hidden_state_mut = Lchainbert_hidden_state_mut[0]
        
        Hchainbert_outputs_org = self.Hchainbert_org(input_ids = H_ids_org,attention_mask = H_attention_mask_org)
        Hchainbert_hidden_state_org = [Hchainbert_outputs_org.hidden_states[i] for i in self.layer]
        if len(Hchainbert_hidden_state_org) > 1 :
            Hchainbert_hidden_state_org = torch.stack(Hchainbert_hidden_state_org,dim=2)
            if self.compound_mode == 'sum':
                Hchainbert_hidden_state_org = torch.sum(Hchainbert_hidden_state_org,dim=2,keepdim=False)
            elif self.compound_mode == 'mean':
                Hchainbert_hidden_state_org = torch.mean(Hchainbert_hidden_state_org,dim=2,keepdim=False)
        else:
            Hchainbert_hidden_state_org = Hchainbert_hidden_state_org[0]
        

        Lchainbert_outputs_org = self.Lchainbert_org(input_ids = L_ids_org,attention_mask = L_attention_mask_org)
        Lchainbert_hidden_state_org = [Lchainbert_outputs_org.hidden_states[i] for i in self.layer]
        if len(Lchainbert_hidden_state_org) > 1 :
            Lchainbert_hidden_state_org = torch.stack(Lchainbert_hidden_state_org,dim=2)
            if self.compound_mode == 'sum':
                Lchainbert_hidden_state_org = torch.sum(Lchainbert_hidden_state_org,dim=2,keepdim=False)
            elif self.compound_mode == 'mean':
                Lchainbert_hidden_state_org = torch.mean(Lchainbert_hidden_state_org,dim=2,keepdim=False)
        else:
            Lchainbert_hidden_state_org = Lchainbert_hidden_state_org[0]
        
        Hchainbert_hidden_state_mut = torch.mean(Hchainbert_hidden_state_mut,dim=1,keepdim=False)
        Lchainbert_hidden_state_mut = torch.mean(Lchainbert_hidden_state_mut,dim=1,keepdim=False)
        Hchainbert_hidden_state_org = torch.mean(Hchainbert_hidden_state_org,dim=1,keepdim=False)
        Lchainbert_hidden_state_org = torch.mean(Lchainbert_hidden_state_org,dim=1,keepdim=False)

        HLchainbert_hidden_state_mut = torch.concat([Hchainbert_hidden_state_mut,Lchainbert_hidden_state_mut,epitope_feature_mut],dim=-1)
        HLchainbert_hidden_state_org = torch.concat([Hchainbert_hidden_state_org,Lchainbert_hidden_state_org,epitope_feature_org],dim=-1)

        HLchainbert_hidden_state_mut = self.ln_mut(HLchainbert_hidden_state_mut)
        HLchainbert_hidden_state_mut = self.fc1_mut(HLchainbert_hidden_state_mut)
        HLchainbert_hidden_state_mut = self.dp1_mut(HLchainbert_hidden_state_mut)
        HLchainbert_hidden_state_mut = self.fc2_mut(HLchainbert_hidden_state_mut)
        HLchainbert_hidden_state_mut = self.dp2_mut(HLchainbert_hidden_state_mut)

        HLchainbert_hidden_state_org = self.ln_org(HLchainbert_hidden_state_org)
        HLchainbert_hidden_state_org = self.fc1_org(HLchainbert_hidden_state_org)
        HLchainbert_hidden_state_org = self.dp1_org(HLchainbert_hidden_state_org)
        HLchainbert_hidden_state_org = self.fc2_org(HLchainbert_hidden_state_org)
        HLchainbert_hidden_state_org = self.dp2_org(HLchainbert_hidden_state_org)

        logits = torch.sum(HLchainbert_hidden_state_mut - HLchainbert_hidden_state_org,dim =-1)
        
        
        return logits