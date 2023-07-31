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
from torch.utils.data import DataLoader, Dataset
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

annotations_df.rename(columns = {'worker_id':'annotator_id', 'comment':'text', 'toxic_score':'annotation'}, inplace = True)


grouped = annotations_df.groupby('sentence_id')['annotation'].nunique().reset_index()
grouped.columns = ['sentence_id', 'unique_annotations']
annotations_df = annotations_df.merge(grouped, on='sentence_id')
annotations_df['disagreement'] = annotations_df['unique_annotations'] > 1


splitter = GroupShuffleSplit(test_size=0.2, n_splits=2, random_state = 2)
split = splitter.split(annotations_df, groups=annotations_df['sentence_id'])
train_inds, test_inds = next(split)
train_df = annotations_df.iloc[train_inds]
test_df = annotations_df.iloc[test_inds]
train_df = train_df.sample(frac=1)
test_df = test_df.sample(frac=1)


# In[6]:


labels = train_df['annotation'].unique()


# In[7]:


#sort labels
labels.sort()


# In[8]:


# embedding_dim = 100


# ## bert + annotator

# In[9]:




device = torch.device('cuda')


# In[11]:


# configuration = BertConfig()
# configuration.num_labels = len(labels)
# configuration.num_annotators = len(total_annotator_ids)
# configuration.annotator_embedding_dim = 100
# configuration.hidden_size = 768 

configuration = BertConfig.from_pretrained("bert-base-uncased")
configuration.num_labels = len(labels)
configuration.num_annotators = len(total_annotator_ids)
configuration.annotator_embedding_dim = 512
configuration.hidden_size = 768 
model = bert.BertForSequenceClassificationWithAnnotators(configuration).to(device)




# In[12]:


# Define batch size and number of workers for data loaders
batch_size = 16
num_workers = 2

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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



# Create training and testing datasets
train_dataset = CustomDataset(train_df, tokenizer)
test_dataset = CustomDataset(test_df, tokenizer)

# Create training and testing data loaders
# train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
# test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)




# bert.train_trainer(model, device, train_data_loader, test_data_loader, mode="annotators", freeze= True)




# torch.save(model.state_dict(), 'annotators.pth')





# bert.test(model, device, test_data_loader, mode="annotators")

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
        
        outputs = model(annotator_ids = annotator_ids, input_ids =input_ids, attention_mask = attention_mask, labels =                     labels, freeze = False)
        loss = outputs[0]
        
        if return_outputs:
#             return loss, outputs
            return loss, {"logits":outputs[1], "disagreement":disagreement} 
        return loss


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)
#     preds = pred.predictions.argmax(-1)
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


# In[11]:

# Define the training arguments

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
    compute_metrics=compute_metrics
)



trainer.train()









