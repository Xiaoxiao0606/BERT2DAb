# BERT2DAb
This repository provides the code for BERT2DAb, a pre-trained model for antibody representation based on amino acid sequences and 2D-structure. This project is done by Information Center, Academy of Miliary Medical Sciences.

Installation

Our project is based on pytorch=1.11.0 (python version = 3.8.0), transformers=4.18.0. The environment can be installed by following instructions:
```
  $ git clone https://github.com/Xiaoxiao0606/BERT2DAb.git
  $ cd BERT2DAb
  $ pip install -r requirements.txt
```


# Download
The pre-trained model and the vocabulary will be downloaded by run the following commands.

BERT2DAb_H:
```
  $  import torch
  $  from transformers import BertTokenizer,BertModel
  $  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  $  tokenizer_H = BertTokenizer.from_pretrained("w139700701/BERT2DAb_H")
  $  model_H = BertModel.from_pretrained("w139700701/BERT2DAb_H")
```

BERT2DAb_L:
```
  $  import torch
  $  from transformers import BertTokenizer,BertModel
  $  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  $  tokenizer_L = BertTokenizer.from_pretrained("w139700701/BERT2DAb_L")
  $  model_L = BertModel.from_pretrained("w139700701/BERT2DAb_L")
```


# Contact Information
For help or suggestions for BERT2DAb, please concact Xiaowei Luo(lxw920701@163.com).




