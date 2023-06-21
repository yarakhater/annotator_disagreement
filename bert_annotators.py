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


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config): 
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size + config.annotator_embedding_dim, config.num_labels)
        self.annotator_embeddings = nn.Embedding(config.num_annotators, config.annotator_embedding_dim)

        self.init_weights()
        
    def forward(
        self,
        annotator_ids=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs,
    ):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]
        annotator_embeddings = self.annotator_embeddings(annotator_ids)

        pooled_output = torch.cat((pooled_output, annotator_embeddings), dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


# In[9]:


device = torch.device('cuda')


# In[10]:


configuration = BertConfig()
configuration.num_labels = len(labels)
configuration.num_annotators = len(total_annotator_ids)
configuration.annotator_embedding_dim = 100
configuration.hidden_size = 768 
model = BertForSequenceClassification(configuration).to(device)


# In[11]:


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


# In[ ]:


# Training loop
get_ipython().run_line_magic('time', '')
num_epochs = 10
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss().to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        annotator_ids = batch['annotator_id'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        outputs = model(annotator_ids = annotator_ids, input_ids =input_ids, attention_mask = attention_mask, labels = labels)
        loss = outputs[0]
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_data_loader)

    print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {average_loss:.4f}')


# In[ ]:





# In[ ]:




