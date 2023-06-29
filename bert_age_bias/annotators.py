#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("..")

from sklearn.model_selection import train_test_split
import pandas as pd
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


# In[2]:


train_df = pd.read_csv('../data/train_older_adult_annotations.csv',delimiter=',', encoding='latin-1')
test_df = pd.read_csv('../data/test_annotations.csv', delimiter=',')
df = pd.concat([test_df, train_df])

age_anxiety_df = pd.read_csv('../data/age_anxiety_full_responses.csv', delimiter=',')
age_experience_df = pd.read_csv('../data/age_experience_responses.csv', delimiter=',')
demographics_df = pd.read_csv('../data/demographics_responses.csv', delimiter=',')
anxiety_score_df = pd.read_csv('../data/respondent_anxiety_table.csv', delimiter=',')

df1 = pd.merge(demographics_df, anxiety_score_df, on='respondent_id')
merged_df = pd.merge(df, df1, on='respondent_id')

sentiment_labels = ['Very negative','Somewhat negative','Neutral','Somewhat positive','Very positive']
total_annotator_ids = merged_df['respondent_id'].unique().tolist()

id2label = {index: row for (index, row) in enumerate(sentiment_labels)} 
label2id = {row: index for (index, row) in enumerate(sentiment_labels)}

id2annotator = {index: row for (index, row) in enumerate(total_annotator_ids)}
annotator2id = {row: index for (index, row) in enumerate(total_annotator_ids)}

merged_df["annotation"] = merged_df["annotation"].map(label2id)
merged_df["respondent_id"] = merged_df["respondent_id"].map(annotator2id)

merged_df.rename(columns = {'respondent_id':'annotator_id', 'unit_text':'text'}, inplace = True)


# In[3]:


splitter = GroupShuffleSplit(test_size=0.3, n_splits=2, random_state = 0)
split = splitter.split(merged_df, groups=merged_df['unit_id'])
train_inds, test_inds = next(split)
train_df = merged_df.iloc[train_inds]
test_df = merged_df.iloc[test_inds]
train_df = train_df.sample(frac=1)
test_df = test_df.sample(frac=1)


# In[4]:


labels = merged_df['annotation'].unique()
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


configuration = BertConfig()
configuration.num_labels = len(labels)
configuration.num_annotators = len(total_annotator_ids)
configuration.annotator_embedding_dim = 100
configuration.hidden_size = 768 
model = bert.BertForSequenceClassificationWithAnnotators(configuration).to(device)


# In[9]:


bert.train(model, device, train_data_loader, mode="annotators")


# In[ ]:


torch.save(model.state_dict(), 'annotators.pth')


# In[8]:


bert.test(model, device, test_data_loader, mode="annotators")


# In[ ]:




