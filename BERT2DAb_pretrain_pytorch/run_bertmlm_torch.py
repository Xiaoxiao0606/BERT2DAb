import argparse
import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from model import BERT
from trainer import BERTTrainer,anti_BERTTrainer
from dataset import BERTDataset,antibody_IterableDataset,antibody_normalDataset,antibody_normalDataset2,WordVocab
from transformers import BertTokenizer,BertConfig, BertForMaskedLM,DataCollatorForLanguageModeling
import torch
import numpy as np
import pandas as pd
import random




def parse(server):
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset_path", 
                        type=str, 
                        default='./data/',
                        help="train dataset for train bert")
    parser.add_argument("-v", "--vocab_path", 
                        type=str, 
                        default='./vocabs/H_wordpiece/vocab.txt',
                        help="built Tokenizer with transformers.BertTokenizer")
    parser.add_argument("-o", "--output_path", 
                        type=str, 
                        default='./models/',
                        help="")

    parser.add_argument("-mt", "--model_type", type=str, default='base', help="base or large")


    parser.add_argument("-b", "--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=8, help="dataloader worker size")
    parser.add_argument("-p", "--prefetch_factor", type=int, default=60, help="prefetch factor")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=100, help="printing loss every n iter: setting n")
    parser.add_argument("--checkpoint_freq", type=int, default=20000, help="checkpoint save freq")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0,1,2,3], help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--warmup_steps", type=int, default=100000, help="warmup-steps")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    return parser



def main():
    args = parser.parse_args()

    args.output_path =args.output_path + f'_BS{args.batch_size}' +f'_EPS{args.epochs}' + f'_LR{args.lr}' +f'_WD{args.adam_weight_decay}'+f'_WS{args.warmup_steps}'+'_num4'

    print(args)

    os.makedirs(f'{args.output_path}/log',exist_ok = True)
    os.makedirs(f'{args.output_path}/runs',exist_ok = True)
    os.makedirs(f'{args.output_path}/model',exist_ok = True)

    torch.cuda.empty_cache()
    
    #DistributedDataParallel
    if args.local_rank  != -1:
        torch.cuda.set_device(args.local_rank )
        device=torch.device("cuda", args.local_rank )
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    print("Loading Tokenizer:", args.vocab_path)
    tokenizer = BertTokenizer(args.vocab_path,do_lower_case=False)
    print("Vocab Size: ", tokenizer.vocab_size)

    
    print("Building BERT model")
    if (args.model_type  == 'base') & (args.first_trained_model_path is None):
            config = BertConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=768, 
                num_hidden_layers=12, 
                num_attention_heads=12,
                max_position_embeddings=128,
                pad_token_id=3,
                type_vocab_size = 1
            )
    elif (args.model_type  == 'large') & (args.first_trained_model_path is None):
            config  = BertConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=1024, 
                num_hidden_layers=24, 
                num_attention_heads=16,
                max_position_embeddings=128,
                pad_token_id=3,
                type_vocab_size = 1
            )

    if args.first_trained_model_path is None:
        bert_model = BertForMaskedLM(config)
        print('No of parameters: ', bert_model.num_parameters())
    else:
        bert_model = BertForMaskedLM.from_pretrained(args.first_trained_model_path,output_hidden_states=True, return_dict=True)
        print('No of parameters: ', bert_model.num_parameters())

    # Initialize the BERT Language Model, with mlm
    bert_model.to(device)

    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        bert_model = nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank ],
                                                    output_device=args.local_rank)

    print("Loading Train Dataset", args.train_dataset_path)
    train_dataset = antibody_normalDataset(file_path = args.train_dataset_path)
    print("Train Dataset Size:",len(train_dataset))
    print(train_dataset.__getitem__(0))

   

    print("Creating Dataloader")
    data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )
    train_sampler = DistributedSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset, 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers,
                                pin_memory=True,
                                persistent_workers=True,
                                collate_fn=data_collator,
                                prefetch_factor = args.prefetch_factor ,
                                sampler = train_sampler
                                )
                              
    
    print("Creating anti_BERT Trainer")
    total_steps = (len(train_dataset) // (args.batch_size * torch.cuda.device_count())) * args.epochs if len(train_dataset) % (args.batch_size * torch.cuda.device_count()) == 0 else (len(train_dataset) // (args.batch_size * torch.cuda.device_count()) + 1) * args.epochs
    warmup_steps = args.warmup_steps
    # print(total_steps,warmup_steps)
    trainer = anti_BERTTrainer(bert_model, 
                               train_dataloader=train_data_loader, 
                               lr= args.lr,
                               betas=(args.adam_beta1, args.adam_beta2), 
                               weight_decay=args.adam_weight_decay,
                               with_cuda=args.with_cuda, 
                               cuda_devices=args.cuda_devices, 
                               log_freq=args.log_freq,
                               local_rank=args.local_rank,
                               output_path = args.output_path,
                               checkpoint_freq = args.checkpoint_freq,
                               warmup_steps = warmup_steps,
                               total_steps =total_steps)

    torch.distributed.barrier()
    
    print("Training Start")
    early_stop = False
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        early_stop = trainer.train(epoch)
        if early_stop == True:
            break
        if args.local_rank == 0 :
            trainer.save_epoch(epoch, args.output_path)
        torch.distributed.barrier()
    print('Over!!!')

if __name__=='__main__':
 
    main()
