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
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch





train_df = pd.read_csv("train.csv", delimiter=',')
test_df = pd.read_csv("test.csv", delimiter=',')
annotators_df = pd.read_csv("annotators.csv", delimiter=',')




total_annotator_ids = annotators_df['annotator_id'].unique().tolist()


labels = train_df['annotation'].unique()



labels.sort()





device = torch.device('cuda')

assigned_groups = {}
entropy = {}




configuration = BertConfig.from_pretrained("bert-base-uncased")
configuration.num_labels = len(labels)
configuration.num_annotators = len(total_annotator_ids)
configuration.group_embedding_dim = 512
configuration.num_groups = 15
configuration.hidden_size = 768 
model = bert.BertForSequenceClassificationWithGroups(configuration).to(device)



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
    remove_unused_columns=False)


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
    compute_metrics=compute_metrics
)



trainer.train()




model.load_state_dict(model.state_dict())
model.eval()
assigned_groups_trained = model.group_assignment.argmax(dim=1)
clusters = assigned_groups_trained.cpu()

print("assigned_groups_trained", len(assigned_groups_trained))


data = pd.DataFrame({'Clusters': clusters, 'Gender': annotators_df['gender']})
grouped_data = data.groupby(['Clusters', 'Gender']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Clusters', columns='Gender', values='Count')
pivot_table.plot(kind='bar', stacked=True)


plt.savefig('plots/gender.png')
plt.close()


data = pd.DataFrame({'Clusters': clusters, 'Identify As Transgender': annotators_df['identify_as_transgender']})
grouped_data = data.groupby(['Clusters', 'Identify As Transgender']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Clusters', columns='Identify As Transgender', values='Count')
pivot_table.plot(kind='bar', stacked=True)


plt.savefig('plots/identify_as_transgender.png')
plt.close()



data = pd.DataFrame({'Clusters': clusters, 'Is Parent': annotators_df['is_parent']})
grouped_data = data.groupby(['Clusters', 'Is Parent']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Clusters', columns='Is Parent', values='Count')
pivot_table.plot(kind='bar', stacked=True)


plt.savefig('plots/is_parent.png')
plt.close()


data = pd.DataFrame({'Clusters': clusters, 'LGBTQ Status': annotators_df['lgbtq_status']})
grouped_data = data.groupby(['Clusters', 'LGBTQ Status']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Clusters', columns='LGBTQ Status', values='Count')
pivot_table.plot(kind='bar', stacked=True)


plt.savefig('plots/lgbtq_status.png')
plt.close()


data = pd.DataFrame({'Clusters': clusters, 'Personally Been Target': annotators_df['personally_been_target']})
grouped_data = data.groupby(['Clusters', 'Personally Been Target']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Clusters', columns='Personally Been Target', values='Count')
pivot_table.plot(kind='bar', stacked=True)

plt.savefig('plots/personally_been_target.png')
plt.close()


data = pd.DataFrame({'Clusters': clusters, 'Personally Seen Toxic Content': annotators_df['personally_seen_toxic_content']})
grouped_data = data.groupby(['Clusters', 'Personally Seen Toxic Content']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Clusters', columns='Personally Seen Toxic Content', values='Count')
pivot_table.plot(kind='bar', stacked=True)

plt.savefig('plots/personally_seen_toxic_content.png')
plt.close()


data = pd.DataFrame({'Clusters': clusters, 'Political Affilation': annotators_df['political_affilation']})
grouped_data = data.groupby(['Clusters', 'Political Affilation']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Clusters', columns='Political Affilation', values='Count')
pivot_table.plot(kind='bar', stacked=True)

plt.savefig('plots/political_affilation.png')
plt.close()


data = pd.DataFrame({'Clusters': clusters, 'Race': annotators_df['race']})
grouped_data = data.groupby(['Clusters', 'Race']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Clusters', columns='Race', values='Count')
pivot_table.plot(kind='bar', stacked=True)

plt.savefig('plots/race.png')
plt.close()


data = pd.DataFrame({'Clusters': clusters, 'Religion Important': annotators_df['religion_important']})
grouped_data = data.groupby(['Clusters', 'Religion Important']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Clusters', columns='Religion Important', values='Count')
pivot_table.plot(kind='bar', stacked=True)

plt.savefig('plots/religion_important.png')
plt.close()

data = pd.DataFrame({'Clusters': clusters, 'Technology Impact': annotators_df['technology_impact']})
grouped_data = data.groupby(['Clusters', 'Technology Impact']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Clusters', columns='Technology Impact', values='Count')
pivot_table.plot(kind='bar', stacked=True)

plt.savefig('plots/technology_impact.png')
plt.close()


data = pd.DataFrame({'Clusters': clusters, 'Toxic Comments Problem': annotators_df['toxic_comments_problem']})
grouped_data = data.groupby(['Clusters', 'Toxic Comments Problem']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Clusters', columns='Toxic Comments Problem', values='Count')
pivot_table.plot(kind='bar', stacked=True)

plt.savefig('plots/toxic_comments_problem.png')
plt.close()

data = pd.DataFrame({'Clusters': clusters, 'Uses Media Forums': annotators_df['uses_media_forums']})
grouped_data = data.groupby(['Clusters', 'Uses Media Forums']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Clusters', columns='Uses Media Forums', values='Count')
pivot_table.plot(kind='bar', stacked=True)

plt.savefig('plots/uses_media_forums.png')
plt.close()

data = pd.DataFrame({'Clusters': clusters, 'Uses Media News': annotators_df['uses_media_news']})
grouped_data = data.groupby(['Clusters', 'Uses Media News']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Clusters', columns='Uses Media News', values='Count')
pivot_table.plot(kind='bar', stacked=True)

plt.savefig('plots/uses_media_news.png')
plt.close()

data = pd.DataFrame({'Clusters': clusters, 'Uses Media Social': annotators_df['uses_media_social']})
grouped_data = data.groupby(['Clusters', 'Uses Media Social']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Clusters', columns='Uses Media Social', values='Count')
pivot_table.plot(kind='bar', stacked=True)

plt.savefig('plots/uses_media_social.png')
plt.close()

data = pd.DataFrame({'Clusters': clusters, 'Uses Media Video': annotators_df['uses_media_video']})
grouped_data = data.groupby(['Clusters', 'Uses Media Video']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Clusters', columns='Uses Media Video', values='Count')
pivot_table.plot(kind='bar', stacked=True)

plt.savefig('plots/uses_media_video.png')
plt.close()






