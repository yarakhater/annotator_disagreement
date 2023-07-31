#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
sys.path.append("..")
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import nltk
from torch import optim
from torch.utils.data import Dataset
from nltk.corpus import stopwords
from models import bert
from transformers import BertTokenizer, DistilBertTokenizer, DistilBertForSequenceClassification, BertPreTrainedModel, BertModel, BertForSequenceClassification, BertConfig, get_constant_schedule, get_linear_schedule_with_warmup
import json
from torch.utils.data import DataLoader
import numpy as np


# In[4]:


train_df = pd.read_csv('train_new_agr.csv',delimiter=',', encoding='latin-1')
test_df = pd.read_csv('test_new_agr.csv', delimiter=',')

total_annotator_ids = train_df['annotator_id'].unique().tolist()


train_labels = train_df['annotation'].unique()
test_labels = test_df['annotation'].unique()
labels = np.unique(np.concatenate((train_labels, test_labels), axis=0))
#sort labels
labels.sort()
num_labels_glob=len(labels)


# In[5]:


device = torch.device("cuda")


# In[6]:

configuration = BertConfig.from_pretrained("bert-base-uncased")
configuration.num_labels = len(labels)
model = bert.BertForSequenceClassificationText(configuration).to(device)
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels_glob)
# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels_glob)
# for param in model.distilbert.parameters():
#     param.requires_grad = False
    
    
# for name, param in model.named_parameters():
#     if 'classifier' not in name: # classifier layer
#         param.requires_grad = False


# In[7]:


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = self.data.iloc[index]['text']
#         annotator_id = self.data.iloc[index]['annotator_id']
        annotation = self.data.iloc[index]['annotation']
        disagreement = self.data.iloc[index]['disagreement']

        # Tokenize the sentence
        inputs = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
#         annotator_id = torch.tensor(annotator_id, dtype=torch.long)
        annotation = torch.tensor(annotation, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
#             'annotator_id': annotator_id,
            'label': annotation,
            'disagreement': disagreement
        }


# In[8]:


# Define batch size and number of workers for data loaders
batch_size = 16
num_workers = 2

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Create training and testing datasets
train_dataset = CustomDataset(train_df, tokenizer)
test_dataset = CustomDataset(test_df, tokenizer)

# Create training and testing data loaders
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)


# In[9]:


from transformers import TrainingArguments
from sklearn.metrics import accuracy_score
from transformers import Trainer


class CustomTrainer_text(Trainer):
    def compute_loss(self, model, inputs, device=torch.device("cuda"), return_outputs=False):
        
        input_ids = inputs.get("input_ids").to(device)
        attention_mask = inputs.get("attention_mask").to(device)
        labels = inputs.get("labels").to(device)
        disagreement = inputs.get("disagreement").to(device)
        
        outputs = model(input_ids =input_ids, attention_mask = attention_mask, labels = labels, freeze=True)
        loss = outputs[0]
        
        if return_outputs:
            return loss, {"logits":outputs[1], "disagreement":disagreement} 
        return loss


# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     acc = accuracy_score(labels, preds)
#     return {
#       'accuracy': acc,
#     }
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)
    acc = accuracy_score(labels, preds)
    disagreement = pred.predictions[1]
    labels_agreement = labels[~disagreement]
    labels_disagreement = labels[disagreement]
    predicted_agreement = preds[~disagreement]
    predicted_disagreement = preds[disagreement] 
    agreement_acc = accuracy_score(labels_agreement, predicted_agreement)
    disagreement_acc = accuracy_score(labels_disagreement, predicted_disagreement)
    return {
      'agreement_accuracy': agreement_acc,
      'disagreement_accuracy':disagreement_acc,
      'accuracy':acc
    }

num_epochs = 10
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay = 0.01)
schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500,             num_training_steps=len(train_dataset)*num_epochs)
optimizers = optimizer, schedule

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=num_epochs,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
#     warmup_steps=500,
#     weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=250,
    evaluation_strategy = "epoch",
    logging_strategy="epoch",
    remove_unused_columns=False,
#     optim= "adamw_torch",
#     learning_rate=0.01,
)





trainer = CustomTrainer_text(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    optimizers = optimizers
)



trainer.train()


