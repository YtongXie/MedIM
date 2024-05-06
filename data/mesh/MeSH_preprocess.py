import re
import torch
import tqdm
# from transformers import GPT2Tokenizer, GPT2Model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2Model.from_pretrained('gpt2')
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

terms = []
meshFile = '/home/ytxie/userdisk1/ytxie/SSL/MedIM/data/mesh/mesh_ACE_pre.txt'
with open(meshFile, mode='rb') as file:
    mesh = file.readlines()

MeSH = []
for line in mesh:
    for sub_line in line.decode('utf-8')[0:-1].lower().split(','):
        if sub_line[0]==' ':
            sub_line = sub_line[1::]
        if sub_line[-1]==' ':
            sub_line = sub_line[0:-1]
        MeSH.append(sub_line)

n = 0
for line in MeSH:
    # text = line.decode('utf-8')[0:-1].lower()
    text = line
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    feats = output[0].sum(0).sum(0)
    terms.append(feats)

    n = n+1
    print(n)

torch.save(terms, '/home/ytxie/SSL/MGCA-main/data/mesh/mesh_ACE_feats.pth')
