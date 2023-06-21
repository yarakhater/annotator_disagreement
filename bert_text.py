#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit 
import matplotlib.pyplot as plt
from models import utils

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, BertPreTrainedModel, BertModel, BertConfig
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from tqdm import tqdm
import time


# ## data prep

# In[2]:


annotations_df = pd.read_csv("data/Toxicity_content/toxic_content_annotation", delimiter=',')
text_df = pd.read_csv("data/Toxicity_content/toxic_content_sentences", delimiter=',')
annotators_df = pd.read_csv("data/Toxicity_content/toxic_content_workers", delimiter=',')


# In[3]:


text_df


# In[4]:


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


# In[5]:


labels = train_df['toxic_score'].unique()


# In[6]:


#sort labels
labels.sort()


# In[7]:


embedding_dim = 100


# ## bert

# In[8]:


device = torch.device('cuda')


# In[9]:


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels)).to(device)


# In[10]:


# Define batch size and number of workers for data loaders
batch_size = 16
num_workers = 2

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create training and testing datasets


train_inputs = tokenizer(train_df['comment'].tolist(), padding=True, truncation=True, return_tensors='pt')
train_labels = torch.tensor(train_df['toxic_score'].tolist())
train_dataset = torch.utils.data.TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_inputs['token_type_ids'], train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size)

test_inputs = tokenizer(test_df['comment'].tolist(), padding=True, truncation=True, return_tensors='pt')
test_labels = torch.tensor(test_df['toxic_score'].tolist())
test_dataset = torch.utils.data.TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_inputs['token_type_ids'], test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# In[11]:


# Training loop
num_epochs = 10
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss().to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['label'].to(device)
        input_ids, attention_mask, token_type_ids, labels = [item.to(device) for item in batch]


        # Forward pass
        outputs = model(input_ids =input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids,labels = labels)
        loss = outputs[0]
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_data_loader)

    print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {average_loss:.4f}')


# In[ ]:




