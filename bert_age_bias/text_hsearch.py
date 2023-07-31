#!/usr/bin/env python
# coding: utf-8

# In[12]:


import sys
sys.path.append("..")
import os
os.environ["RAY_FUNCTION_SIZE_ERROR_THRESHOLD"] = "5000"  # Set a higher threshold value

from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit 
import matplotlib.pyplot as plt
import nltk
from torch import optim
from nltk.corpus import stopwords
# from models import bert
from transformers import BertTokenizer, BertForSequenceClassification, BertPreTrainedModel, BertModel
import json
from torch.utils.data import DataLoader, Dataset
import ray
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from functools import partial

print(os.environ["RAY_FUNCTION_SIZE_ERROR_THRESHOLD"])

ray.init(ignore_reinit_error=True, num_cpus=3)
print("success")



class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, labels):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = self.data.iloc[index]['text']
        annotator_id = self.data.iloc[index]['annotator_id']
        annotation = self.data.iloc[index]['annotation']

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

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'annotator_id': annotator_id,
            'label': annotation
        }





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


# In[14]:


splitter = GroupShuffleSplit(test_size=0.3, n_splits=2, random_state = 0)
split = splitter.split(merged_df, groups=merged_df['unit_id'])
train_inds, test_inds = next(split)
train_df = merged_df.iloc[train_inds]
test_val_df = merged_df.iloc[test_inds]
splitter = GroupShuffleSplit(test_size=0.5, n_splits=2, random_state = 0)
split = splitter.split(test_val_df, groups=test_val_df['unit_id'])
val_inds, test_inds = next(split)
val_df = test_val_df.iloc[val_inds]
test_df = test_val_df.iloc[test_inds]
train_df = train_df.sample(frac=1)
test_df = test_df.sample(frac=1)    
val_df = val_df.sample(frac=1)
len(train_df["annotator_id"].unique()), len(val_df["annotator_id"].unique()), len(test_df["annotator_id"].unique())


# In[15]:


labels = merged_df['annotation'].unique()
#sort labels
labels.sort()


# In[16]:


device = torch.device("cuda")


# In[17]:


# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels)).to(device)
# for param in model.base_model.parameters():
#     param.requires_grad = False


# In[18]:


# Define batch size and number of workers for data loaders
batch_size = 8
num_workers = 2

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create training and testing datasets
train_dataset = CustomDataset(train_df, tokenizer, labels)
val_dataset = CustomDataset(val_df, tokenizer, labels)
test_dataset = CustomDataset(test_df, tokenizer, labels)

# Create training and testing data loaders
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)


# In[19]:


config = {
    "lr": tune.choice([0.000001, 0.00001, 0.0001, 0.001, 0.0]),
    "weight_decay": tune.choice([0.001, 0.01, 0.1, 0.2])
}


# In[20]:


mode = "text"


# In[21]:


def train(config):
    # Training loop
    labels = merged_df['annotation'].unique()
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels)).to(device)
    print("modeeeel",model)
    for param in model.base_model.parameters():
        param.requires_grad = False
    num_epochs = 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay = config["weight_decay"])
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    checkpoint = session.get_checkpoint()
    
    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0


    for epoch in range(start_epoch, num_epochs):
        print("epochh:", epoch)
        model.train()
        total_loss = 0
        for batch in train_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            annotator_ids = batch['annotator_id'].to(device)
            labels = batch['label'].to(device)
            if mode=="groups" :
                w, log_p = model(annotator_ids = annotator_ids, input_ids =input_ids, attention_mask = attention_mask, labels =                   labels)
                loss = torch.zeros(input_ids.size(0))
                for i in range(input_ids.size(0)):
                    loss[i] = - (w[i].log_softmax(dim=1) + log_p[i].reshape(-1, 1)).logsumexp(dim=0)[labels[i]]
                # Backward pass and optimization
                loss = torch.mean(loss)
            elif mode=="annotators":
                # Forward pass
                outputs = model(annotator_ids = annotator_ids, input_ids =input_ids, attention_mask = attention_mask, labels =                     labels)
                loss = outputs[0]
            else:
                outputs = model(input_ids =input_ids, attention_mask = attention_mask, labels = labels)
                loss = outputs[0]
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_data_loader)
        
        total_val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for batch in val_data_loader:
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                annotator_ids = batch['annotator_id'].to(device)
                labels = batch['label'].to(device)
                if mode=="groups" :
                    w, log_p = model(annotator_ids = annotator_ids, input_ids =input_ids, attention_mask = attention_mask, labels = labels)
                    loss = torch.zeros(input_ids.size(0))
                    for i in range(input_ids.size(0)):
                        loss[i] = - (w[i].log_softmax(dim=1) + log_p[i].reshape(-1, 1)).logsumexp(dim=0)[labels[i]]
                    loss = torch.mean(loss)
                    best_group = log_p.argmax(dim=1)
                    w = w[range(len(w)), best_group]
                    _, predicted = torch.max(w, 1)
                elif mode=="annotators":
                    # Forward pass
                    outputs = model(annotator_ids = annotator_ids, input_ids =input_ids, attention_mask = attention_mask, labels = labels)
                    loss = outputs[0]
                    logits = outputs[1]
                    # Get predicted labels
                    _, predicted = torch.max(logits, dim=1)
                else:
                    outputs = model(input_ids =input_ids, attention_mask = attention_mask, labels = labels)
                    loss = outputs[0]
                    logits = outputs[1]
                    # Get predicted labels
                    _, predicted = torch.max(logits, dim=1)
                    
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()


                total_val_loss += loss.item()
    
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
        average_val_loss = total_val_loss / len(val_data_loader)
        accuracy = correct/ total

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"loss": average_val_loss, "accuracy": accuracy},
            checkpoint=checkpoint,
        )

        print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {average_loss:.4f}')


# In[ ]:


scheduler = ASHAScheduler(
       metric="loss",
       mode="min",
       max_t=10,
       grace_period=1,
       reduction_factor=2,
   )
print(os.environ["RAY_FUNCTION_SIZE_ERROR_THRESHOLD"])
result = tune.run(
   partial(train),
   resources_per_trial={"cpu": 3, "gpu": 1},
   config=config,
   num_samples=10,
   scheduler=scheduler
)


print(result)

# # In[30]:


# # train(model, device, train_data_loader,val_data_loader, mode="text")


# # In[ ]:


# torch.save(model.state_dict(), 'text.pth')


# # In[ ]:


# bert.test(model, device, test_data_loader, mode="text")


# # In[ ]:




