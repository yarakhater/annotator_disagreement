#!/usr/bin/env python
# coding: utf-8


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
from transformers import BertTokenizer, DistilBertForSequenceClassification, BertPreTrainedModel, BertModel, BertConfig, get_linear_schedule_with_warmup
import json
from torch.utils.data import DataLoader
import numpy as np
from models import bert




train_df = pd.read_csv('train_new_agr.csv',delimiter=',', encoding='latin-1')
test_df = pd.read_csv('test_new_agr.csv', delimiter=',')

total_annotator_ids = train_df['annotator_id'].unique().tolist()


train_labels = train_df['annotation'].unique()
test_labels = test_df['annotation'].unique()
labels = np.unique(np.concatenate((train_labels, test_labels), axis=0))

labels.sort()
num_labels_glob=len(labels)



device = torch.device("cuda")



configuration = BertConfig.from_pretrained("bert-base-uncased")
configuration.num_labels = len(labels)
configuration.num_annotators = len(total_annotator_ids)
configuration.annotator_embedding_dim = 512
configuration.hidden_size = 768 
model = bert.BertForSequenceClassificationWithAnnotators(configuration).to(device)



class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = self.data.iloc[index]['text']
        annotator_id = self.data.iloc[index]['annotator_id']
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
        annotator_id = torch.tensor(annotator_id, dtype=torch.long)
        annotation = torch.tensor(annotation, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'annotator_ids': annotator_id,
            'label': annotation,
            'disagreement': disagreement
        }



batch_size = 16

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create training and testing datasets
train_dataset = CustomDataset(train_df, tokenizer)
test_dataset = CustomDataset(test_df, tokenizer)



from transformers import TrainingArguments
from sklearn.metrics import accuracy_score

from transformers import Trainer

class CustomTrainer_annotators_text(Trainer):
    def compute_loss(self, model, inputs, device=torch.device("cuda"), return_outputs=False):
        
        input_ids = inputs.get("input_ids").to(device)
        attention_mask = inputs.get("attention_mask").to(device)
        annotator_ids = inputs.get("annotator_ids").to(device)
        labels = inputs.get("labels").to(device)
        disagreement = inputs.get("disagreement").to(device)
        
        outputs = model(annotator_ids = annotator_ids, input_ids =input_ids, attention_mask = attention_mask, labels =                     labels, freeze = True)
        loss = outputs[0]
        
        if return_outputs:
            return loss, {"logits":outputs[1], "disagreement":disagreement} 
        return loss


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




training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=250,
    evaluation_strategy = "epoch",
    logging_strategy="epoch",
    remove_unused_columns=False,
    optim="adamw_torch"
)




trainer = CustomTrainer_annotators_text(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)



trainer.train()


