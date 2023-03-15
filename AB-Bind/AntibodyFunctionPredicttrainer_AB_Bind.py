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

class AntibodyFunctionPredicttrainer_forregression(object):
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
        

        
        self.loss = nn.MSELoss(reduction='sum')

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
        res=self.iteration(epoch, val_data,lab,train=False)
        return res

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

        pre_list_train = []
        label_list_train =[]

        batch_acc_avg_val = []
        batch_F1Score_avg_val = []
        batch_precision_avg_val = []
        batch_recall_avg_val = []
        batch_f1_avg_val = []

        
        
        for i, data in data_iter:
            data = {key: value.cuda() for key, value in data.items()}
            
            
            
            outputs_logits = self.model(H_ids_mut = data["H_ids_mut"], H_attention_mask_mut = data["H_attention_mask_mut"],L_ids_mut = data['L_ids_mut'],L_attention_mask_mut = data['L_attention_mask_mut'],
                                        H_ids_org = data["H_ids_org"], H_attention_mask_org = data["H_attention_mask_org"],L_ids_org = data['L_ids_org'],L_attention_mask_org = data['L_attention_mask_org'],
                                        epitope_feature_mut = data['epitope_feature_mut'],epitope_feature_org = data['epitope_feature_org']
                                       )
            loss = self.loss(outputs_logits.float(), data['label'].float()).cuda()

            
            if train:
                
                self.optim.zero_grad() 
                loss.backward()
                self.optim.step()
                self.optim_schedule.step()
            
    
            if label == 'test':
                avg_loss += loss.item()
            else:
                reduced_loss = self.reduce_tensor(loss.data)
                self.avg_loss += reduced_loss.item()
                avg_loss += reduced_loss.item()
                
                
                
                gather_predicts = [torch.zeros_like(outputs_logits) for _ in range(dist.get_world_size())]
                dist.all_gather(gather_predicts,outputs_logits)
                gather_predicts = torch.concat(gather_predicts,dim = 0)
                gather_label = [torch.zeros_like(data['label']) for _ in range(dist.get_world_size())]
                dist.all_gather(gather_label,data['label'])
                gather_label = torch.concat(gather_label,dim = 0)
                

            if (label == 'validation') & (self.local_rank == 0):
                batch_rmse = metrics.mean_squared_error(gather_label.cpu().numpy(), gather_predicts.cpu().numpy())
                batch_Pearson,batch_Pvalue = stats.pearsonr(gather_label.cpu().numpy(), gather_predicts.cpu().numpy())
            elif (label == 'test') & (self.local_rank == 0):
                batch_rmse = metrics.mean_squared_error(data['label'].cpu().numpy(), outputs_logits.cpu().numpy())
                batch_Pearson,batch_Pvalue = stats.pearsonr(data['label'].cpu().numpy(), outputs_logits.cpu().numpy())
            elif (label == 'train') & (self.local_rank == 0):
                batch_rmse = metrics.mean_squared_error(gather_label.cpu().numpy(), gather_predicts.cpu().numpy())
                batch_Pearson,batch_Pvalue = stats.pearsonr(gather_label.cpu().numpy(), gather_predicts.cpu().numpy())


            if (label == 'test') & (self.local_rank == 0):
                pre_list_test.extend(outputs_logits.cpu().numpy().tolist())
                label_list_test.extend(data['label'].cpu().numpy().tolist())
            elif (label == 'validation') & (self.local_rank == 0):
                pre_list_val.extend(gather_predicts.cpu().numpy().tolist())
                label_list_val.extend(gather_label.cpu().numpy().tolist())
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
                    "step_RMSE":batch_rmse,
                    "step_Pearson":batch_Pearson,
                    "step_P":batch_Pvalue
                }
            elif (label == 'validation') & (self.local_rank == 0) :
                post_fix = {
                    "state":label,
                    "fold":self.fold,
                    "epoch": epoch,
                    "iter":  i,
                    "epoch_avg_loss":avg_loss/(i+1),
                    "step_loss": reduced_loss.item(),
                    "step_RMSE":batch_rmse,
                    "step_Pearson":batch_Pearson,
                    "step_P":batch_Pvalue,
                    "sample_RMSE_collect_validation": metrics.mean_squared_error(label_list_val, pre_list_val),
                    "sample_Pearson_collect_validation":stats.pearsonr(label_list_val, pre_list_val)[0],
                    "sample_P_collect_validation":stats.pearsonr(label_list_val, pre_list_val)[1],

                }
            elif (label == 'test') & (self.local_rank == 0):
                post_fix = {
                    "state":label,
                    "epoch": epoch,
                    "iter": i,
                    "epoch_avg_loss":avg_loss/(i+1),
                    "step_loss": loss.item(),
                    "step_RMSE":batch_rmse,
                    "step_Pearson":batch_Pearson,
                    "step_P":batch_Pvalue,
                    "sample_RMSE_collect_test": metrics.mean_squared_error(label_list_test, pre_list_test),
                    "sample_Pearson_collect_test":stats.pearsonr(label_list_test, pre_list_test)[0],
                    "sample_P_collect_test":stats.pearsonr(label_list_test, pre_list_test)[1],

                }

            niter = epoch * len(data_iter) + (i)
            if (niter % self.log_freq == 0) & (label == 'train') & (self.local_rank == 0):
                data_iter.write(str(post_fix))
                self.writer.add_scalar(f'{str(self.fold)}/Train/avg_loss', self.avg_loss / (niter + 1), niter)
                self.writer.add_scalar(f'{str(self.fold)}/Train/step_loss',reduced_loss.item(), niter)
                self.writer.add_scalar(f'{str(self.fold)}/Train/step_RMSE',batch_rmse, niter)
                self.writer.add_scalar(f'{str(self.fold)}/Train/step_Pearson',batch_Pearson, niter)
                self.writer.add_scalar(f'{str(self.fold)}/Train/step_P',batch_Pvalue, niter)

                
            if (i % self.log_freq == 0) and (label == 'validation')& (self.local_rank == 0):
                data_iter.write(str(post_fix))
                self.writer.add_scalar(f'{str(self.fold)}/validation/step_loss', reduced_loss.item(), niter)
                self.writer.add_scalar(f'{str(self.fold)}/validation/step_RMSE',batch_rmse, niter)
                self.writer.add_scalar(f'{str(self.fold)}/validation/step_Pearson',batch_Pearson, niter)
                self.writer.add_scalar(f'{str(self.fold)}/validation/step_P',batch_Pvalue, niter)
                self.writer.add_scalar(f'{str(self.fold)}/validation/sample_RMSE_collect_validation',metrics.mean_squared_error(label_list_val, pre_list_val), niter)
                self.writer.add_scalar(f'{str(self.fold)}/validation/sample_Pearson_collect_validation', stats.pearsonr(label_list_val, pre_list_val)[0], niter)
                self.writer.add_scalar(f'{str(self.fold)}/validation/sample_P_collect_validation',stats.pearsonr(label_list_val, pre_list_val)[1], niter)
    

            if (i % self.log_freq == 0) & (label == 'test') & (self.local_rank == 0):
                data_iter.write(str(post_fix))
                self.writer.add_scalar('test/step_loss', loss.item(), niter)
                self.writer.add_scalar('test/step_RMSE',batch_rmse, niter)
                self.writer.add_scalar('test/step_Pearson',batch_Pearson, niter)
                self.writer.add_scalar('test/step_P',batch_Pvalue, niter)
                self.writer.add_scalar('test/sample_RMSE_collect_test',metrics.mean_squared_error(label_list_test, pre_list_test), niter)
                self.writer.add_scalar('test/sample_Pearson_collect_test', stats.pearsonr(label_list_test, pre_list_test)[0], niter)
                self.writer.add_scalar('test/sample_P_collect_test',stats.pearsonr(label_list_test, pre_list_test)[1], niter)

            

        if  (label == 'train')  & (self.local_rank == 0):
            self.writer.add_scalar(f'{str(self.fold)}/Train/epoch_avg_loss', avg_loss/len(data_iter), epoch)
            self.writer.add_scalar(f'{str(self.fold)}/Train/sample_RMSE_collect_train',metrics.mean_squared_error(label_list_train, pre_list_train), epoch)
            self.writer.add_scalar(f'{str(self.fold)}/Train/sample_Pearson_collect_train', stats.pearsonr(label_list_train, pre_list_train)[0], epoch)
            self.writer.add_scalar(f'{str(self.fold)}/Train/sample_P_collect_train',stats.pearsonr(label_list_train, pre_list_train)[1], epoch)
    

  
        if (label == 'validation') & (self.local_rank == 0):
            return {'fold':self.fold,
                    'epoch':epoch,
                    f'sample_RMSE_collect_{label}':metrics.mean_squared_error(label_list_val, pre_list_val),
                    f'sample_Pearson_collect_{label}':stats.pearsonr(label_list_val, pre_list_val)[0],
                    f'sample_P_collect_{label}':stats.pearsonr(label_list_val, pre_list_val)[1],
                    }

        if (label == 'test') & (self.local_rank == 0):
            return {'epoch':epoch,
                    f'sample_RMSE_collect_{label}':metrics.mean_squared_error(label_list_test, pre_list_test),
                    f'sample_Pearson_collect_{label}':stats.pearsonr(label_list_test, pre_list_test)[0],
                    f'sample_P_collect_{label}':stats.pearsonr(label_list_test, pre_list_test)[1],
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