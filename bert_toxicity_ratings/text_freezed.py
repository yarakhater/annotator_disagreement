#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("..")

import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit 
import matplotlib.pyplot as plt
from models import bert

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, BertPreTrainedModel, BertModel, BertConfig
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from tqdm import tqdm
import time



train_df = pd.read_csv("train.csv", delimiter=',')
test_df = pd.read_csv("test.csv", delimiter=',')
annotators_df = pd.read_csv("annotators.csv", delimiter=',')




total_annotator_ids = annotators_df['annotator_id'].unique().tolist()


labels = train_df['annotation'].unique()


labels.sort()




device = torch.device('cuda')


configuration = BertConfig.from_pretrained("bert-base-uncased")
configuration.num_labels = len(labels)
configuration.hidden_size = 768 

model = bert.BertForSequenceClassificationText(configuration).to(device)


batch_size = 16
num_workers = 2

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = bert.CustomDataset(train_df, tokenizer, labels)
test_dataset = bert.CustomDataset(test_df, tokenizer, labels)




from transformers import TrainingArguments
from sklearn.metrics import accuracy_score
from transformers import Trainer


class CustomTrainer_text(Trainer):
    def compute_loss(self, model, inputs, device=torch.device("cuda"), return_outputs=False):
        
        input_ids = inputs.get("input_ids").to(device)
        attention_mask = inputs.get("attention_mask").to(device)
        labels = inputs.get("labels").to(device)
        disagreement = inputs.get("disagreement").to(device)
        
        outputs = model(input_ids =input_ids, attention_mask = attention_mask, labels = labels, freeze = True)
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





trainer = CustomTrainer_text(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)



trainer.train()









