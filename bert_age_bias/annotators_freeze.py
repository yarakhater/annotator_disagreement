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
from transformers import BertTokenizer, BertForSequenceClassification, BertPreTrainedModel, BertModel, BertConfig
import json
from torch.utils.data import DataLoader



train_df = pd.read_csv('train_new_agr.csv',delimiter=',', encoding='latin-1')
test_df = pd.read_csv('test_new_agr.csv', delimiter=',')

total_annotator_ids = train_df['annotator_id'].unique().tolist()


train_labels = train_df['annotation'].unique()
test_labels = test_df['annotation'].unique()
labels = np.unique(np.concatenate((train_labels, test_labels), axis=0))
#sort labels
labels.sort()


# In[ ]:





# In[5]:


# Define batch size and number of workers for data loaders
batch_size = 16
num_workers = 2

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create training and testing datasets
train_dataset = bert.CustomDataset(train_df, tokenizer, labels)
test_dataset = bert.CustomDataset(test_df, tokenizer, labels)

# Create training and testing data loaders
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)


# In[6]:


device = torch.device("cuda")


# In[7]:


configuration = BertConfig.from_pretrained("bert-base-uncased")
configuration.num_labels = len(labels)
configuration.num_annotators = len(total_annotator_ids)
configuration.annotator_embedding_dim = 512
configuration.hidden_size = 768 
model = bert.BertForSequenceClassificationWithAnnotators(configuration).to(device)


# In[9]:


bert.train(model, device, train_data_loader, mode="annotators", freeze = True)


# In[ ]:


torch.save(model.state_dict(), 'annotators_freeze.pth')


# In[8]:


# train_accuracy_disagreement, train_accuracy_agreement = bert.get_accuracy(model, device, train_data_loader, mode="annotators")
# test_accuracy_disagreement, test_accuracy_agreement = bert.get_accuracy(model, device, test_data_loader, mode="annotators")

# print("Train Accuracy Disagreement: ", train_accuracy_disagreement) 
# print("Train Accuracy Agreement: ", train_accuracy_agreement) 
# print("Test Accuracy Disagreement: ", test_accuracy_disagreement) 
# print("Test Accuracy Agreement: ", test_accuracy_agreement) 

train_accuracy, train_accuracy_disagreement, train_accuracy_agreement = bert.get_accuracy(model, device, train_data_loader, mode="annotators")
test_accuracy, test_accuracy_disagreement, test_accuracy_agreement = bert.get_accuracy(model, device, test_data_loader, mode="annotators")

print("Train Accuracy : ", train_accuracy) 
print("Train Accuracy Disagreement: ", train_accuracy_disagreement) 
print("Train Accuracy Agreement: ", train_accuracy_agreement) 
print("Test Accuracy : ", test_accuracy) 
print("Test Accuracy Disagreement: ", test_accuracy_disagreement) 
print("Test Accuracy Agreement: ", test_accuracy_agreement) 



# In[ ]:




