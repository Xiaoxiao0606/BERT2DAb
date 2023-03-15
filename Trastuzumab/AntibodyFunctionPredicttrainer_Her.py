from unittest import result
import torch
from torch import nn
from torch.optim import AdamW,Adam
from tqdm import tqdm
import torchmetrics
from numpy import *
from transformers.optimization import get_linear_schedule_with_warmup,get_constant_schedule_with_warmup
import torch.distributed as dist
from sklearn import metrics
from scipy import stats
import torch.nn.functional as F


class AntibodyFunctionPredicttrainer(object):
    def __init__(self,
                 args,
                 model,
                 total_steps=None,
                 writer = None,
                 warmup_steps:int = 100, 
                 betas=(0.9, 0.999), 
                 fold = None,
              
                 ):


        self.model = model
        

        
       
        self.optim = AdamW(self.model.parameters(), lr=args.lr, betas=betas, weight_decay=args.adam_weight_decay)
       
        self.optim_schedule = get_linear_schedule_with_warmup(self.optim,warmup_steps,total_steps)
        

      
        self.writer = writer

        self.log_freq = args.log_freq

        self.local_rank =args.local_rank

       

        self.fold = fold

        self.avg_loss = 0.0
        self.total_correct = 0
        self.total_element = 0
        
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch,train_data,lab = 'train'):
        self.iteration(epoch, train_data,lab)

    def validation(self, epoch,val_data,lab = 'validation'):
        res,label_list_val,logits_list_val_sigmoid=self.iteration(epoch, val_data,lab,train=False)
        return res,label_list_val,logits_list_val_sigmoid

    def test(self, epoch,test_data,lab = 'test'):
        res=self.iteration(epoch, test_data,lab, train=False)
        return res
    
    def reduce_tensor(self,tensor: torch.Tensor):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= torch.cuda.device_count()
        return rt

    def reduce_tensor_sum(self,tensor: torch.Tensor):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        
        return rt

    def iteration(self, epoch, data_loader,label, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every epoch
        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        
        if label =='train':
            self.model.train()
        else:
            self.model.eval()


        str_code = label

        
        data_iter = tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        batch_acc_avg_test = []
        batch_F1Score_avg_test = []
        batch_precision_avg_test = []
        batch_recall_avg_test = []
        batch_f1_avg_test = []

        pre_list_test = []
        label_list_test =[]

        pre_list_val = []
        label_list_val =[]
        logits_list_val=[]

        pre_list_train = []
        label_list_train =[]

        batch_acc_avg_val = []
        batch_F1Score_avg_val = []
        batch_precision_avg_val = []
        batch_recall_avg_val = []
        batch_f1_avg_val = []

        
        
        for i, data in data_iter:
            data = {key: value.cuda() for key, value in data.items()}
            
            # 1. forward
            outputs_logits_sigmoid,resuls_analysis = self.model(H_ids = data["H_ids"], H_attention_mask = data["H_attention_mask"],L_ids = data['L_ids'],L_attention_mask = data['L_attention_mask'],epitope_feature = data['epitope_feature'],H_region=data['H_region'],L_region=data['L_region'])
            predicts = (outputs_logits_sigmoid > 0.5)
            
            # 2. loss
            loss = F.binary_cross_entropy(outputs_logits_sigmoid,  data['label'].float())
            # 3. backward and optimization only in train
            if train:
               
                self.optim.zero_grad() 
                loss.backward()
                self.optim.step()
                self.optim_schedule.step()
            
    
            if label == 'test':
                avg_loss += loss.item()
                correct = predicts.eq(data['label']).sum().item()
                element = data['label'].nelement()
            else:
                correct = predicts.eq(data['label']).sum().item()
                reduced_loss = self.reduce_tensor(loss.data)
               
                reduced_correct = self.reduce_tensor_sum(torch.tensor(correct).cuda())
                reduced_element = self.reduce_tensor_sum(torch.tensor(data['label'].nelement()).cuda())
                self.avg_loss += reduced_loss.item()
                self.total_correct += reduced_correct.item()
                self.total_element += reduced_element.item()
                avg_loss += reduced_loss.item()
                total_correct += reduced_correct
                total_element += reduced_element
                
                
                gather_predicts = [torch.zeros_like(predicts) for _ in range(dist.get_world_size())]
                dist.all_gather(gather_predicts,predicts)
                gather_predicts = torch.concat(gather_predicts,dim = 0)

                gather_label = [torch.zeros_like(data['label']) for _ in range(dist.get_world_size())]
                dist.all_gather(gather_label,data['label'])
                gather_label = torch.concat(gather_label,dim = 0)

                gather_logits = [torch.zeros_like(outputs_logits_sigmoid) for _ in range(dist.get_world_size())]
                dist.all_gather(gather_logits,outputs_logits_sigmoid)
                gather_logits = torch.concat(gather_logits,dim = 0)
                
                
                

            if (label == 'validation') & (self.local_rank == 0):
                batch_acc = metrics.accuracy_score(gather_label.cpu(),gather_predicts.cpu())
                batch_F1Score = metrics.f1_score(gather_label.cpu(),gather_predicts.cpu(),zero_division=0)
                batch_precision = metrics.precision_score(gather_label.cpu(),gather_predicts.cpu(),zero_division=0)
                batch_recall = metrics.recall_score(gather_label.cpu(),gather_predicts.cpu(),zero_division=0)
            elif (label == 'test') & (self.local_rank == 0):
                batch_acc = metrics.accuracy_score(data['label'].cpu(),predicts.cpu())
                batch_F1Score = metrics.f1_score(data['label'].cpu(),predicts.cpu(),zero_division=0)
                batch_precision = metrics.precision_score(data['label'].cpu(),predicts.cpu(),zero_division=0)
                batch_recall = metrics.recall_score(data['label'].cpu(),predicts.cpu(),zero_division=0)
            elif (label == 'train') & (self.local_rank == 0):
                batch_acc = metrics.accuracy_score(gather_label.cpu(),gather_predicts.cpu())
                batch_F1Score = metrics.f1_score(gather_label.cpu(),gather_predicts.cpu(),zero_division=0)
                batch_precision = metrics.precision_score(gather_label.cpu(),gather_predicts.cpu(),zero_division=0)
                batch_recall = metrics.recall_score(gather_label.cpu(),gather_predicts.cpu(),zero_division=0)


            if (label == 'test') & (self.local_rank == 0):
                
                pre_list_test.extend(predicts.cpu().numpy().tolist())
                label_list_test.extend(data['label'].cpu().numpy().tolist())
            elif (label == 'validation') & (self.local_rank == 0):
                pre_list_val.extend(gather_predicts.cpu().numpy().tolist())
                label_list_val.extend(gather_label.cpu().numpy().tolist())
                logits_list_val.extend(gather_logits.cpu().numpy().tolist())
            elif (label == 'train') & (self.local_rank == 0):
                pre_list_train.extend(gather_predicts.cpu().numpy().tolist())
                label_list_train.extend(gather_label.cpu().numpy().tolist())

                
            if (label == 'train') & (self.local_rank == 0) :
                post_fix = {
                    "state":label,
                    "fold":self.fold,
                    "epoch": epoch,
                    "niter": epoch * len(data_iter) + i,
                    "avg_loss": self.avg_loss /(epoch * len(data_iter) + (i+1)),
                    "step_loss": reduced_loss.item(),
                    "epoch_avg_loss":avg_loss/(i+1),
                    "step_acc":round((reduced_correct.item() / reduced_element.item()) * 100,2),
                    "step_acc_tm": batch_acc,
                    "step_F1Score_tm":batch_F1Score,
                    "step_precision_tm":batch_precision,
                    "step_recall_tm":batch_recall,
                }
            elif (label == 'validation') & (self.local_rank == 0) :
                post_fix = {
                    "state":label,
                    "fold":self.fold,
                    "epoch": epoch,
                    "iter":  i,
                    "epoch_avg_loss":avg_loss/(i+1),
                    "step_loss": reduced_loss.item(),
                    "step_acc":(reduced_correct.item() / reduced_element.item()) * 100,
                    "step_acc_tm": batch_acc,
                    "step_F1Score_tm":batch_F1Score,
                    "step_precision_tm":batch_precision,
                    "step_recall_tm":batch_recall,
                    "sample_acc_tm_collect_validation":metrics.accuracy_score(torch.Tensor(label_list_val).cpu().int(),torch.Tensor(pre_list_val).cpu().int()),
                    "sample_F1Score_tm_collect_validation":metrics.f1_score(torch.Tensor(label_list_val).cpu().int(),torch.Tensor(pre_list_val).cpu().int(),zero_division=0),
                    "sample_precision_tm_collect_validation":metrics.precision_score(torch.Tensor(label_list_val).cpu().int(),torch.Tensor(pre_list_val).cpu().int(),zero_division=0),
                    "sample_recall_tm_collect_validation":metrics.recall_score(torch.Tensor(label_list_val).cpu().int(),torch.Tensor(pre_list_val).cpu().int(),zero_division=0),

                }
            elif (label == 'test') & (self.local_rank == 0):
                post_fix = {
                    "state":label,
                    "epoch": epoch,
                    "iter": i,
                    "epoch_avg_loss":avg_loss/(i+1),
                    "step_loss": loss.item(),
                    "step_acc":(correct / element) * 100,
                    "step_acc_tm": batch_acc,
                    "step_F1Score_tm":batch_F1Score,
                    "step_precision_tm":batch_precision,
                    "step_recall_tm":batch_recall,
                    "sample_acc_tm_collect_test":metrics.accuracy_score(torch.Tensor(label_list_test).cpu().int(),torch.Tensor(pre_list_test).cpu().int()),
                    "sample_F1Score_tm_collect_test":metrics.f1_score(torch.Tensor(label_list_test).cpu().int(),torch.Tensor(pre_list_test).cpu().int(),zero_division=0),
                    "sample_precision_tm_collect_test":metrics.precision_score(torch.Tensor(label_list_test).cpu().int(),torch.Tensor(pre_list_test).cpu().int(),zero_division=0),
                    "sample_recall_tm_collect_test":metrics.recall_score(torch.Tensor(label_list_test).cpu().int(),torch.Tensor(pre_list_test).cpu().int(),zero_division=0),

                }

            niter = epoch * len(data_iter) + (i)
            if (niter % self.log_freq == 0) & (label == 'train') & (self.local_rank == 0):
                data_iter.write(str(post_fix))
                self.writer.add_scalar(f'{str(self.fold)}/Train/avg_loss', self.avg_loss / (niter + 1), niter)
                self.writer.add_scalar(f'{str(self.fold)}/Train/step_loss',reduced_loss.item(), niter)
                self.writer.add_scalar(f'{str(self.fold)}/Train/step_acc',reduced_correct.item() / reduced_element.item(), niter)
                self.writer.add_scalar(f'{str(self.fold)}/Train/step_F1',batch_F1Score, niter)

                
            if (i % self.log_freq == 0) and (label == 'validation') & (self.local_rank == 0):
                data_iter.write(str(post_fix))
                self.writer.add_scalar(f'{str(self.fold)}/validation/step_loss', reduced_loss.item(), niter)
                self.writer.add_scalar(f'{str(self.fold)}/validation/step_acc', (reduced_correct.item() / reduced_element.item()) * 100, niter)
                self.writer.add_scalar(f'{str(self.fold)}/validation/step_acc_tm',batch_acc, niter)
                self.writer.add_scalar(f'{str(self.fold)}/validation/step_F1Score_tm', batch_F1Score, niter)
                self.writer.add_scalar(f'{str(self.fold)}/validation/step_precision_tm',batch_precision, niter)
                self.writer.add_scalar(f'{str(self.fold)}/validation/step_recall_tm', batch_recall, niter)
                self.writer.add_scalar(f'{str(self.fold)}/validation/sample_acc_tm_collect_validation',metrics.accuracy_score(torch.Tensor(label_list_val).cpu().int(),torch.Tensor(pre_list_val).cpu().int()), niter)
                self.writer.add_scalar(f'{str(self.fold)}/validation/sample_F1Score_tm_collect_validation', metrics.f1_score(torch.Tensor(label_list_val).cpu().int(),torch.Tensor(pre_list_val).cpu().int(),zero_division=0), niter)
                self.writer.add_scalar(f'{str(self.fold)}/validation/sample_precision_tm_collect_validation',metrics.precision_score(torch.Tensor(label_list_val).cpu().int(),torch.Tensor(pre_list_val).cpu().int(),zero_division=0), niter)
                self.writer.add_scalar(f'{str(self.fold)}/validation/sample_recall_tm_collect_validation', metrics.recall_score(torch.Tensor(label_list_val).cpu().int(),torch.Tensor(pre_list_val).cpu().int(),zero_division=0), niter)

            if (i % self.log_freq == 0) & (label == 'test') & (self.local_rank == 0):
                data_iter.write(str(post_fix))
                self.writer.add_scalar('test/step_loss', loss.item(), niter)
                self.writer.add_scalar('test/step_acc', (correct / element) * 100, niter)
                self.writer.add_scalar('test/step_acc_tm',batch_acc, niter)
                self.writer.add_scalar('test/step_F1Score_tm', batch_F1Score, niter)
                self.writer.add_scalar('test/step_precision_tm',batch_precision, niter)
                self.writer.add_scalar('test/step_recall_tm', batch_recall, niter)
                self.writer.add_scalar('test/sample_acc_tm_collect_test',metrics.accuracy_score(torch.Tensor(label_list_test).cpu().int(),torch.Tensor(pre_list_test).cpu().int()), niter)
                self.writer.add_scalar('test/sample_F1Score_tm_collect_test', metrics.f1_score(torch.Tensor(label_list_test).cpu().int(),torch.Tensor(pre_list_test).cpu().int(),zero_division=0), niter)
                self.writer.add_scalar('test/sample_precision_tm_collect_test',metrics.precision_score(torch.Tensor(label_list_test).cpu().int(),torch.Tensor(pre_list_test).cpu().int(),zero_division=0), niter)
                self.writer.add_scalar('test/sample_recall_tm_collect_test', metrics.recall_score(torch.Tensor(label_list_test).cpu().int(),torch.Tensor(pre_list_test).cpu().int(),zero_division=0), niter)

            

        if  (label == 'train')  & (self.local_rank == 0):
            self.writer.add_scalar(f'{str(self.fold)}/Train/epoch_avg_loss', avg_loss/len(data_iter), epoch)
            self.writer.add_scalar(f'{str(self.fold)}/Train/epoch_acc', (total_correct/total_element) * 100, epoch)
            self.writer.add_scalar(f'{str(self.fold)}/Train/epoch_F1', metrics.f1_score(torch.Tensor(label_list_train).cpu().int(),torch.Tensor(pre_list_train).cpu().int(),zero_division=0), epoch)
        

  
        if label == 'validation':
            results = {'fold':self.fold,
                    'epoch':epoch,
                    f'sample_acc_tm_collect_{label}':metrics.accuracy_score(torch.Tensor(label_list_val).cpu().int(),torch.Tensor(pre_list_val).cpu().int()),
                    f'sample_F1Score_tm_collect_{label}':metrics.f1_score(torch.Tensor(label_list_val).int(),torch.Tensor(pre_list_val).cpu().int(),zero_division=0),
                    f'sample_precision_tm_collect_{label}':metrics.precision_score(torch.Tensor(label_list_val).cpu().int(),torch.Tensor(pre_list_val).cpu().int(),zero_division=0),
                    f'sample_recall_tm_collect_{label}':metrics.recall_score(torch.Tensor(label_list_val).cpu().int(),torch.Tensor(pre_list_val).cpu().int(),zero_division=0),
                    }

            return results,label_list_val,logits_list_val 

        if (label == 'test') & (self.local_rank == 0):
            return {'epoch':epoch,
                    f'sample_acc_tm_collect_{label}':metrics.accuracy_score(torch.Tensor(label_list_test).cpu().int(),torch.Tensor(pre_list_test).cpu().int()),
                    f'sample_F1Score_tm_collect_{label}':metrics.f1_score(torch.Tensor(label_list_test).cpu().int(),torch.Tensor(pre_list_test).cpu().int(),zero_division=0),
                    f'sample_precision_tm_collect_{label}':metrics.precision_score(torch.Tensor(label_list_test).cpu().int(),torch.Tensor(pre_list_test).cpu().int(),zero_division=0),
                    f'sample_recall_tm_collect_{label}':metrics.recall_score(torch.Tensor(label_list_test).cpu().int(),torch.Tensor(pre_list_test).cpu().int(),zero_division=0),
                    } 
        
        

    def save(self, epoch, file_path):
        """
        Saving the current model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".pkl"
        
        if torch.cuda.device_count() > 1:
            torch.save(self.model.module, output_path)
        elif torch.cuda.device_count() == 1:
            torch.save(self.model, output_path)
        print("EP:%d Model Saved on:%s" % (epoch, output_path))
        return output_path
