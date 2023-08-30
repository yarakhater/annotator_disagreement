#!/usr/bin/env python
# coding: utf-8


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
from transformers import BertTokenizer, BertForSequenceClassification, BertPreTrainedModel, BertModel, BertConfig, get_linear_schedule_with_warmup
import json
from torch.utils.data import DataLoader, Dataset





train_df = pd.read_csv('train_new_agr.csv',delimiter=',', encoding='latin-1')
test_df = pd.read_csv('test_new_agr.csv', delimiter=',')

demographics_df = pd.read_csv('demographics.csv', delimiter=',')
anxiety_score_df = pd.read_csv('anxiety_score.csv', delimiter=',')


total_annotator_ids = train_df['annotator_id'].unique().tolist()


train_labels = train_df['annotation'].unique()
test_labels = test_df['annotation'].unique()
labels = np.unique(np.concatenate((train_labels, test_labels), axis=0))



#sort labels
labels.sort()
num_labels_glob=len(labels)

device = torch.device("cuda")

assigned_groups = {}
entropy = {}



configuration = BertConfig.from_pretrained("bert-base-uncased")
configuration.num_labels = len(labels)
configuration.num_annotators = len(total_annotator_ids)
configuration.group_embedding_dim = 512
configuration.num_groups = 15
configuration.hidden_size = 768 
model = bert.BertForSequenceClassificationWithGroups(configuration).to(device)

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
# num_workers = 2

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    

# Create training and testing datasets
train_dataset = CustomDataset(train_df, tokenizer)
test_dataset = CustomDataset(test_df, tokenizer)



    
from transformers import TrainingArguments
from sklearn.metrics import accuracy_score



def compute_metrics(pred):
    labels = pred.label_ids
    w = torch.tensor(pred.predictions[0])
    log_p = torch.tensor(pred.predictions[1])
    disagreement = pred.predictions[2]
    annotator_ids = pred.predictions[3]
    p = torch.exp(log_p)
    entropy_ar = - torch.sum(p*log_p, axis=1)
    best_group = log_p.argmax(dim=1)
    
    for annotator_id, ent in zip(annotator_ids, entropy_ar):
        ent = float(ent.item())
        if annotator_id in entropy:
            entropy[annotator_id] = list(set(entropy[annotator_id] + [ent]))
        else:
            entropy[annotator_id] = [ent]

    
    for annotator_id, bg in zip(annotator_ids, best_group):
        bg = int(bg.item())
        if annotator_id in assigned_groups:
            assigned_groups[annotator_id] = list(set(assigned_groups[annotator_id] + [bg]))
        else:
            assigned_groups[annotator_id] = [bg]
        
    
    w = w[range(len(w)), best_group]
    _, preds = torch.max(w, 1)
    acc = accuracy_score(labels, preds)
    
    
    labels_agreement = labels[~disagreement]
    labels_disagreement = labels[disagreement]
    predicted_agreement = preds[~disagreement]
    predicted_disagreement = preds[disagreement] 
    agreement_acc = accuracy_score(labels_agreement, predicted_agreement)
    disagreement_acc = accuracy_score(labels_disagreement, predicted_disagreement)
    
    return {
      'agreement_accuracy': agreement_acc,
      'disagreement_accuracy':disagreement_acc,
      'accuracy': acc,
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
    optim="adamw_torch",
    evaluation_strategy = "epoch",
    logging_strategy="epoch",
    remove_unused_columns=False
)


from transformers import Trainer

class CustomTrainer_groups_text(Trainer):
    def compute_loss(self, model, inputs, device=torch.device("cuda"), return_outputs=False):
        

        input_ids = inputs.get("input_ids").to(device)
        attention_mask = inputs.get("attention_mask").to(device)
        annotator_ids = inputs.get("annotator_ids").to(device)
        labels = inputs.get("labels").to(device)
        disagreement = inputs.get("disagreement").to(device)
        

        w, log_p = model(annotator_ids = annotator_ids, input_ids =input_ids, attention_mask = attention_mask, labels = labels,           freeze = False)
        loss = torch.zeros(input_ids.size(0))
        for i in range(input_ids.size(0)):
            loss[i] = - (w[i].log_softmax(dim=1) + log_p[i].reshape(-1, 1)).logsumexp(dim=0)[labels[i].long()]
        loss = torch.mean(loss)

        
        if return_outputs:
            return loss, {'w': w, 'log_p': log_p, "disagreement":disagreement, "annotator_ids":annotator_ids}
        return loss





trainer = CustomTrainer_groups_text(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)



trainer.train()





model.load_state_dict(model.state_dict())
model.eval()
assigned_groups_trained = model.group_assignment.argmax(dim=1)
clusters = assigned_groups_trained.cpu()

print("assigned_groups_trained", len(assigned_groups_trained))



bins = [20, 30, 40, 50, 60, 70, 80, 90, 100]

anxiety_score_df["anxiety_level"] = pd.cut(anxiety_score_df["anxiety_score"], bins=bins)


data = pd.DataFrame({'Cluster': clusters, 'Anxiety Level': anxiety_score_df["anxiety_level"]})
grouped_data = data.groupby(['Cluster', 'Anxiety Level']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Anxiety Level', values='Count')
pivot_table.plot(kind='bar', stacked=True)


plt.savefig('plots/anxiety_level.png')

plt.close()




data = pd.DataFrame({'Cluster': clusters, 'Age Group': demographics_df.iloc[:, 1]})
grouped_data = data.groupby(['Cluster', 'Age Group']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Age Group', values='Count')
pivot_table.plot(kind='bar', stacked=True)


plt.savefig('plots/age_group.png')

plt.close()


data = pd.DataFrame({'Cluster': clusters, 'Race Group': demographics_df.iloc[:, 2]})
grouped_data = data.groupby(['Cluster', 'Race Group']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Race Group', values='Count')
pivot_table.plot(kind='bar', stacked=True)


plt.savefig('plots/race_group.png')

plt.close()


data = pd.DataFrame({'Cluster': clusters, 'Hespanic/Latino': demographics_df.iloc[:, 4]})
grouped_data = data.groupby(['Cluster', 'Hespanic/Latino']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Hespanic/Latino', values='Count')
pivot_table.plot(kind='bar', stacked=True)


plt.savefig('plots/hespanic_latino.png')

plt.close()


data = pd.DataFrame({'Cluster': clusters, 'Area where raised': demographics_df.iloc[:, 5]})
grouped_data = data.groupby(['Cluster', 'Area where raised']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Area where raised', values='Count')
pivot_table.plot(kind='bar', stacked=True)


plt.savefig('plots/area_raised.png')

plt.close()


data = pd.DataFrame({'Cluster': clusters, 'Current Area': demographics_df.iloc[:, 6]})
grouped_data = data.groupby(['Cluster', 'Current Area']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Current Area', values='Count')
pivot_table.plot(kind='bar', stacked=True)


plt.savefig('plots/current_area.png')

plt.close()


data = pd.DataFrame({'Cluster': clusters, 'Current Region': demographics_df.iloc[:, 7]})
grouped_data = data.groupby(['Cluster', 'Current Region']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Current Region', values='Count')
pivot_table.plot(kind='bar', stacked=True)


plt.savefig('plots/current_region.png')

plt.close()


data = pd.DataFrame({'Cluster': clusters, 'Annual Income': demographics_df.iloc[:, 8]})
grouped_data = data.groupby(['Cluster', 'Annual Income']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Annual Income', values='Count')
pivot_table.plot(kind='bar', stacked=True)


plt.savefig('plots/annual_income.png')

plt.close()


data = pd.DataFrame({'Cluster': clusters, 'Education': demographics_df.iloc[:, 9]})
grouped_data = data.groupby(['Cluster', 'Education']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Education', values='Count')
pivot_table.plot(kind='bar', stacked=True)


plt.savefig('plots/education.png')

plt.close()


data = pd.DataFrame({'Cluster': clusters, 'Employment': demographics_df.iloc[:, 10]})
grouped_data = data.groupby(['Cluster', 'Employment']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Employment', values='Count')
pivot_table.plot(kind='bar', stacked=True)

plt.savefig('plots/employment.png')

plt.close()

data = pd.DataFrame({'Cluster': clusters, 'Living Situation': demographics_df.iloc[:, 11]})
grouped_data = data.groupby(['Cluster', 'Living Situation']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Living Situation', values='Count')
pivot_table.plot(kind='bar', stacked=True)

plt.savefig('plots/living_situation.png')

plt.close()


data = pd.DataFrame({'Cluster': clusters, 'Political Identification': demographics_df.iloc[:, 13]})
grouped_data = data.groupby(['Cluster', 'Political Identification']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Political Identification', values='Count')
pivot_table.plot(kind='bar', stacked=True)

plt.savefig('plots/political_identification.png')


plt.close()

data = pd.DataFrame({'Cluster': clusters, 'Gender': demographics_df.iloc[:, 14]})
grouped_data = data.groupby(['Cluster', 'Gender']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Gender', values='Count')
pivot_table.plot(kind='bar', stacked=True)

plt.savefig('plots/gender.png')

plt.close()


