#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("..")

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit 
import matplotlib.pyplot as plt
import nltk
from torch import optim
from nltk.corpus import stopwords
from models import bert
from transformers import BertTokenizer, BertForSequenceClassification, BertPreTrainedModel, BertModel, BertConfig, DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
import json
from torch.utils.data import DataLoader


# In[2]:


train_df = pd.read_csv('train_new_agr.csv',delimiter=',', encoding='latin-1')
test_df = pd.read_csv('test_new_agr.csv', delimiter=',')

total_annotator_ids = train_df['annotator_id'].unique().tolist()


train_labels = train_df['annotation'].unique()
test_labels = test_df['annotation'].unique()
labels = np.unique(np.concatenate((train_labels, test_labels), axis=0))
#sort labels
labels.sort()


device = torch.device("cuda")


configuration = BertConfig.from_pretrained("bert-base-uncased")
# configuration = DistilBertConfig.from_pretrained("distilbert-base-uncased")
configuration.num_labels = len(labels)
configuration.num_annotators = len(total_annotator_ids)
configuration.annotator_embedding_dim = 300
configuration.hidden_size = 768 

model = bert.BertForSequenceClassificationText(configuration).to(device)
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels)).to(device)
# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(labels)).to(device)



# for param in model.base_model.parameters():
#     param.requires_grad = False

# for name, param in model.named_parameters():
#     if 'classifier' not in name: # classifier layer
#         param.requires_grad = False


# Define batch size and number of workers for data loaders
batch_size = 16
num_workers = 2

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Create training and testing datasets
train_dataset = bert.CustomDataset(train_df, tokenizer, labels)
test_dataset = bert.CustomDataset(test_df, tokenizer, labels)

# Create training and testing data loaders
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)



bert.train(model, device, train_data_loader, mode="text", freeze = False)
# bert.train_trainer(model, device, train_data_loader, test_data_loader, mode="text", freeze = True)



# torch.save(model.state_dict(), 'text.pth')

# train_accuracy = bert.get_accuracy(model, device, train_data_loader, mode="text")
# test_accuracy = bert.get_accuracy(model, device, test_data_loader, mode="text")

# print("Train Accuracy : ", train_accuracy) 
# print("Test Accuracy : ", test_accuracy) 



train_accuracy, train_accuracy_disagreement, train_accuracy_agreement = bert.get_accuracy(model, device, train_data_loader, mode="text")
test_accuracy, test_accuracy_disagreement, test_accuracy_agreement = bert.get_accuracy(model, device, test_data_loader, mode="text")

print("Train Accuracy : ", train_accuracy) 
print("Train Accuracy Disagreement: ", train_accuracy_disagreement) 
print("Train Accuracy Agreement: ", train_accuracy_agreement) 
print("Test Accuracy : ", test_accuracy) 
print("Test Accuracy Disagreement: ", test_accuracy_disagreement) 
print("Test Accuracy Agreement: ", test_accuracy_agreement) 