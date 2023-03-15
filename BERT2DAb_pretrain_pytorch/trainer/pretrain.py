import os
import sys
import time
sys.path.append(os.path.abspath('..'))
import torch
import torch.nn as nn
from torch.optim import Adam,AdamW
from torch.utils.data import DataLoader
from model import BERTLM, BERT
from .optim_schedule import ScheduledOptim
from .EarlyStopping import EarlyStopping
from transformers.optimization import get_linear_schedule_with_warmup
import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class anti_BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, model, 
                 train_dataloader: DataLoader, 
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps:int=None,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10,local_rank: int =-1,
                 output_path:str=None,checkpoint_freq:int = 10000,total_steps:int=None):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """
        
        '''nn.DataParallel
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        # cuda_condition = torch.cuda.is_available() and with_cuda
        # self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # Distributed GPU training if CUDA can detect more than 1 GPU
        # if with_cuda and torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for BERT" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        '''
        self.model = model

        # Setting the train and test data loader
        self.train_data = train_dataloader
        # self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        # self.optim = Adam(self.model.parameters(), lr=lr, betas=betas)
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        # self.optim_schedule = ScheduledOptim(self.optim, config.hidden_size, n_warmup_steps=warmup_steps)#from pytorch_bert
        self.optim_schedule = get_linear_schedule_with_warmup(self.optim,warmup_steps,total_steps)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        if local_rank == 0 :
            self.writer = SummaryWriter(log_dir=os.path.join(output_path,
                                                             'runs',datetime.strftime(datetime.now(),'%Y-%m-%d_%H:%M:%S')),
                                                             comment='AntibodyPretrainedModel')

        self.local_rank = local_rank
        self.checkpoint_freq = checkpoint_freq
        self.output_path = output_path

        self.early_stop = EarlyStopping(patience_step=50, min_mean_loss=0.1)

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)
    
    def reduce_tensor(self,tensor: torch.Tensor):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= torch.cuda.device_count()
        return rt


    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        
        self.model.train()

            
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0

        

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.cuda() for key, value in data.items()}
            # 1. forward the next_sentence_prediction and masked_lm model
            outputs = self.model.forward(input_ids = data["input_ids"], attention_mask=data["attention_mask"],labels = data["labels"])
            # 2.Loss of predicting masked token word
            loss = outputs.loss.cuda()
            
            # 3. backward and optimization only in train
            self.optim.zero_grad() 
            loss.backward()
            self.optim.step()
            self.optim_schedule.step()
        
            torch.distributed.barrier()
           
            reduced_loss = self.reduce_tensor(loss.data)
            avg_loss  += reduced_loss.item()
            
            niter = epoch * len(data_iter) + i

            if (i % self.log_freq == 0 ) & (self.local_rank == 0):
                post_fix = {
                "state":str_code,
                "epoch": epoch,
                "niter": niter,
                "avg_loss": avg_loss / (i + 1),
                "loss": reduced_loss.item()
                }

                data_iter.write(str(post_fix))

                self.writer.add_scalar("train/batch_loss_niter", reduced_loss.item(),niter)
                self.writer.add_scalar("train/avg_loss_niter", avg_loss / (i + 1),niter)
                
            if  (niter % self.checkpoint_freq == 0) & (self.local_rank == 0):
                path =  self.output_path + '/log'
                if len(os.listdir(path)) > 4 : 
                    os.chdir(path)
                    os.system(f'ls {path} -tr | head -1 | xargs rm -r')
                    self.save_checkpoint(epoch=epoch,niter=niter, file_path=self.output_path)
                else:
                    self.save_checkpoint(epoch=epoch,niter=niter, file_path=self.output_path)

            torch.distributed.barrier()


          
            self.early_stop(reduced_loss.item())
            if self.early_stop.early_stop:
                if self.local_rank == 0 :
                    r = self.save_earlystop(epoch=epoch,niter=niter, file_path=self.output_path)
            
                torch.distributed.barrier()

                if self.local_rank == 0 :
                    return r
                else:
                    return True
            

        if self.local_rank == 0:
            self.writer.add_scalar("train/avg_loss_epoch",avg_loss / len(data_iter),epoch+1)
            print("EP%d_%s, avg_loss_epoch=" % (epoch, str_code), avg_loss / len(data_iter))
        
        torch.distributed.barrier()
        if self.local_rank == 0:
            return False
        else:
            return


    def save_epoch(self, epoch, file_path):
        output_path_1 = file_path +'/model'+'/AntibodyPretrainedModel'+"_ep%d" % epoch
        output_path_2 = file_path +'/model'+'/AntibodyPretrainedModel'+"_ep%d" % epoch + "/state_dict.pth"
        self.model.module.save_pretrained(output_path_1)
        state = {'model_state_dict':self.model.module.state_dict(), 
                 'optimizer':self.optim.state_dict(), 
                 'optim_schedule':self.optim_schedule.state_dict(),
                 'epoch':epoch,
                 }
        torch.save(state, output_path_2)
        print("EP:%d Model Saved on:" % epoch, output_path_1)
        return output_path_1
    
    def save_earlystop(self, epoch, niter, file_path):
        output_path_1 = file_path +'/model'+'/AntibodyPretrainedModel'+"_ep%d" % epoch+"_step%d" % niter#+".pth"
        output_path_2 = file_path +'/model'+'/AntibodyPretrainedModel'+"_ep%d" % epoch+"_step%d" % niter+"/state_dict.pth"
        self.model.module.save_pretrained(output_path_1)
        state = {'model_state_dict':self.model.module.state_dict(), 
                 'optimizer':self.optim.state_dict(), 
                 'optim_schedule':self.optim_schedule.state_dict(),
                 'epoch':epoch,
                 'niter':niter
                 }
        torch.save(state, output_path_2)
        print("Earlystop EP:%d Model Saved on:" % epoch, output_path_1)
        return True

    def save_checkpoint(self, epoch, niter, file_path):
        output_path_1 = file_path +'/log'+'/AntibodyPretrainedModel'+"_ep%d" % epoch+"_step%d" % niter#+".pth"
        output_path_2 = file_path +'/log'+'/AntibodyPretrainedModel'+"_ep%d" % epoch+"_step%d" % niter+"/state_dict.pth"
        self.model.module.save_pretrained(output_path_1)
        state = {'model_state_dict':self.model.module.state_dict(), 
                 'optimizer':self.optim.state_dict(), 
                 'optim_schedule':self.optim_schedule.state_dict(),
                 'epoch':epoch,
                 'niter':niter
                 }
        torch.save(state, output_path_2)

