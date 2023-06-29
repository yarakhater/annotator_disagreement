#!/usr/bin/env python
# coding: utf-8

# In[2]:

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



# ## data prep

# In[3]:


annotations_df = pd.read_csv("../data/Toxicity_content/toxic_content_annotation", delimiter=',')
text_df = pd.read_csv("../data/Toxicity_content/toxic_content_sentences", delimiter=',')
annotators_df = pd.read_csv("../data/Toxicity_content/toxic_content_workers", delimiter=',')


# In[5]:


annotations_df["comment"] = annotations_df["sentence_id"].map(text_df.set_index("sentence_id")["comment"])
annotations_df["gender"] = annotations_df["worker_id"].map(annotators_df.set_index("worker_id")["gender"])

x = annotations_df.groupby('sentence_id').agg({'toxic_score': lambda x: list(x)})
#keep only sentences that have more than 1 unique annotation in annotations_df
# x = x[x['toxic_score'].apply(lambda x: len(set(x))) > 1]
# annotations_df = annotations_df[annotations_df['sentence_id'].isin(x.index)]

annotators_df = annotators_df[annotators_df['worker_id'].isin(annotations_df['worker_id'])]
print(len(annotators_df))

total_annotator_ids = annotators_df['worker_id'].unique().tolist()
id2annotator = {index: row for (index, row) in enumerate(total_annotator_ids)}
annotator2id = {row: index for (index, row) in enumerate(total_annotator_ids)}
annotations_df["worker_id"] = annotations_df["worker_id"].map(annotator2id)
annotators_df["worker_id"] = annotators_df["worker_id"].map(annotator2id)

splitter = GroupShuffleSplit(test_size=0.2, n_splits=2, random_state = 2)
split = splitter.split(annotations_df, groups=annotations_df['sentence_id'])
train_inds, test_inds = next(split)
train_df = annotations_df.iloc[train_inds]
test_df = annotations_df.iloc[test_inds]
train_df = train_df.sample(frac=1)
test_df = test_df.sample(frac=1)


# In[6]:


labels = train_df['toxic_score'].unique()


# In[7]:


#sort labels
labels.sort()


# In[8]:


embedding_dim = 100


# ## bert + annotator

# In[9]:




device = torch.device('cuda')


# In[11]:


configuration = BertConfig()
configuration.num_labels = len(labels)
configuration.num_annotators = len(total_annotator_ids)
configuration.annotator_embedding_dim = 100
configuration.hidden_size = 768 



# In[12]:


# Define batch size and number of workers for data loaders
batch_size = 16
num_workers = 2

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create training and testing datasets
train_dataset = utils.CustomDataset(train_df, tokenizer, labels)
test_dataset = utils.CustomDataset(test_df, tokenizer, labels)

# Create training and testing data loaders
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

model = bert.BertForSequenceClassificationWithAnnotators(configuration).to(device)


# In[12]:


# Training 

bert.train(model, device, train_data_loader, mode="annotators")

# In[ ]:


torch.save(model.state_dict(), 'annotators.pth')


# In[14]:


bert.test(model, device, test_data_loader, mode="annotators")

# In[ ]:




