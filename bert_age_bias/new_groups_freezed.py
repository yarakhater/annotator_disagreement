#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
sys.path.append("..")
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import nltk
from torch import optim
from torch.utils.data import Dataset
from nltk.corpus import stopwords
from transformers import BertTokenizer, DistilBertForSequenceClassification, BertPreTrainedModel, BertModel, BertConfig, get_linear_schedule_with_warmup
import json
from torch.utils.data import DataLoader
import numpy as np
from models import bert


# In[4]:


train_df = pd.read_csv('train_new_agr.csv',delimiter=',', encoding='latin-1')
test_df = pd.read_csv('test_new_agr.csv', delimiter=',')
# train_df = pd.read_csv('train_new.csv',delimiter=',', encoding='latin-1')
# test_df = pd.read_csv('test_new.csv', delimiter=',')

total_annotator_ids = train_df['annotator_id'].unique().tolist()


train_labels = train_df['annotation'].unique()
test_labels = test_df['annotation'].unique()
labels = np.unique(np.concatenate((train_labels, test_labels), axis=0))
#sort labels
labels.sort()
num_labels_glob=len(labels)


# In[5]:


device = torch.device("cuda")


# In[6]:


# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels_glob)

configuration = BertConfig.from_pretrained("bert-base-uncased")
configuration.num_labels = len(labels)
configuration.num_annotators = len(total_annotator_ids)
configuration.group_embedding_dim = 512
configuration.num_groups = 8
configuration.hidden_size = 768 
model = bert.BertForSequenceClassificationWithGroups(configuration).to(device)

# for param in model.bert.parameters():
#     param.requires_grad = False

# for name, param in model.named_parameters():
#     if 'classifier' not in name: # classifier layer
#         param.requires_grad = False


# In[7]:


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


# In[8]:


# Define batch size and number of workers for data loaders
batch_size = 16
# num_workers = 2

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create training and testing datasets
train_dataset = CustomDataset(train_df, tokenizer)
test_dataset = CustomDataset(test_df, tokenizer)

# Create training and testing data loaders
# train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
# test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)



from transformers import TrainingArguments
from sklearn.metrics import accuracy_score

from transformers import Trainer

class CustomTrainer_groups_text(Trainer):
    def compute_loss(self, model, inputs, device=torch.device("cuda"), return_outputs=False):
        

        input_ids = inputs.get("input_ids").to(device)
        attention_mask = inputs.get("attention_mask").to(device)
        annotator_ids = inputs.get("annotator_ids").to(device)
        labels = inputs.get("labels").to(device)
        disagreement = inputs.get("disagreement").to(device)
        

        w, log_p = model(annotator_ids = annotator_ids, input_ids =input_ids, attention_mask = attention_mask, labels = labels,           freeze = True)
        loss = torch.zeros(input_ids.size(0))
        for i in range(input_ids.size(0)):
            loss[i] = - (w[i].log_softmax(dim=1) + log_p[i].reshape(-1, 1)).logsumexp(dim=0)[labels[i].long()]
        loss = torch.mean(loss)

        
        if return_outputs:
            return loss, {'w': w, 'log_p': log_p, "disagreement":disagreement}
#             return loss, {'w': w, 'log_p': log_p}
#             return loss, w
        return loss


def compute_metrics(pred):
    labels = pred.label_ids
    w = torch.tensor(pred.predictions[0])
    log_p = torch.tensor(pred.predictions[1])
    disagreement = pred.predictions[2]
    best_group = log_p.argmax(dim=1)
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


# In[11]:

num_epochs = 10
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay = 0.01)
schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500,             num_training_steps=len(train_dataset)*num_epochs)
optimizers = optimizer, schedule

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=num_epochs,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
#     warmup_steps=500,
#     weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=250,
    evaluation_strategy = "epoch",
    logging_strategy="epoch",
    remove_unused_columns=False,
#     optim= "adamw_torch",
#     learning_rate=0.01,
)


# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=10,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=250,
#     evaluation_strategy = "epoch",
#     logging_strategy="epoch",
#     remove_unused_columns=False,
#     optim="adamw_torch"
# )




trainer = CustomTrainer_groups_text(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    optimizers = optimizers    
)



trainer.train()





































# #!/usr/bin/env python
# # coding: utf-8

# # In[1]:


# import sys
# sys.path.append("..")

# from sklearn.model_selection import train_test_split
# import pandas as pd
# import numpy as np
# import torch
# from sklearn.model_selection import GroupShuffleSplit 
# import matplotlib.pyplot as plt
# import nltk
# from torch import optim
# from nltk.corpus import stopwords
# from models import bert
# from transformers import BertTokenizer, BertForSequenceClassification, BertPreTrainedModel, BertModel, BertConfig
# import json
# from torch.utils.data import DataLoader, Dataset


# # In[2]:



# train_df = pd.read_csv('train_new_agr.csv',delimiter=',', encoding='latin-1')
# test_df = pd.read_csv('test_new_agr.csv', delimiter=',')

# total_annotator_ids = train_df['annotator_id'].unique().tolist()


# train_labels = train_df['annotation'].unique()
# test_labels = test_df['annotation'].unique()
# labels = np.unique(np.concatenate((train_labels, test_labels), axis=0))



# #sort labels
# labels.sort()
# num_labels_glob=len(labels)

# device = torch.device("cuda")


# configuration = BertConfig.from_pretrained("bert-base-uncased")
# configuration.num_labels = len(labels)
# configuration.num_annotators = len(total_annotator_ids)
# configuration.group_embedding_dim = 128
# configuration.num_groups = 8
# configuration.hidden_size = 768 
# model = bert.BertForSequenceClassificationWithGroups(configuration).to(device)

# for param in model.bert.parameters():
#     param.requires_grad = False

# # for name, param in model.named_parameters():
# #     if 'classifier' not in name: # classifier layer
# #         param.requires_grad = False

# class CustomDataset(Dataset):
#     def __init__(self, dataframe, tokenizer):
#         self.data = dataframe
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         sentence = self.data.iloc[index]['text']
#         annotator_id = self.data.iloc[index]['annotator_id']
#         annotation = self.data.iloc[index]['annotation']
#         disagreement = self.data.iloc[index]['disagreement']

#         # Tokenize the sentence
#         inputs = self.tokenizer.encode_plus(
#             sentence,
#             add_special_tokens=True,
#             padding='max_length',
#             truncation=True,
#             max_length=512,
#             return_tensors='pt'
#         )

#         input_ids = inputs['input_ids'].squeeze()
#         attention_mask = inputs['attention_mask'].squeeze()
#         annotator_id = torch.tensor(annotator_id, dtype=torch.long)
#         annotation = torch.tensor(annotation, dtype=torch.long)

#         return {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'annotator_ids': annotator_id,
#             'label': annotation,
#             'disagreement': disagreement
#         }




# # Define batch size and number of workers for data loaders
# batch_size = 16
# # num_workers = 2

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    

# # Create training and testing datasets
# train_dataset = CustomDataset(train_df, tokenizer)
# test_dataset = CustomDataset(test_df, tokenizer)



    
# from transformers import TrainingArguments
# from sklearn.metrics import accuracy_score



# def compute_metrics(pred):
#     labels = pred.label_ids
#     w = torch.tensor(pred.predictions[0])
#     log_p = torch.tensor(pred.predictions[1])
#     disagreement = pred.predictions[2]
#     best_group = log_p.argmax(dim=1)
#     w = w[range(len(w)), best_group]
#     _, preds = torch.max(w, 1)
#     acc = accuracy_score(labels, preds)
    
#     labels_agreement = labels[~disagreement]
#     labels_disagreement = labels[disagreement]
#     predicted_agreement = preds[~disagreement]
#     predicted_disagreement = preds[disagreement] 
#     agreement_acc = accuracy_score(labels_agreement, predicted_agreement)
#     disagreement_acc = accuracy_score(labels_disagreement, predicted_disagreement)
    
#     return {
#       'agreement_accuracy': agreement_acc,
#       'disagreement_accuracy':disagreement_acc,
#       'accuracy': acc,
#     }


# # In[11]:


# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=10,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=250,
#     optim="adamw_torch",
#     evaluation_strategy = "epoch",
#     logging_strategy="epoch",
#     remove_unused_columns=False,
# )


# from transformers import Trainer

# class CustomTrainer_groups_text(Trainer):
#     def compute_loss(self, model, inputs, device=torch.device("cuda"), return_outputs=False):
        

#         input_ids = inputs.get("input_ids").to(device)
#         attention_mask = inputs.get("attention_mask").to(device)
#         annotator_ids = inputs.get("annotator_ids").to(device)
#         labels = inputs.get("labels").to(device)
#         disagreement = inputs.get("disagreement").to(device)
        

#         w, log_p = model(annotator_ids = annotator_ids, input_ids =input_ids, attention_mask = attention_mask, labels = labels,           freeze = False)
#         loss = torch.zeros(input_ids.size(0))
#         for i in range(input_ids.size(0)):
#             loss[i] = - (w[i].log_softmax(dim=1) + log_p[i].reshape(-1, 1)).logsumexp(dim=0)[labels[i].long()]
#         loss = torch.mean(loss)

        
#         if return_outputs:
#             return loss, {'w': w, 'log_p': log_p, "disagreement":disagreement}
# #             return loss, w
#         return loss





# trainer = CustomTrainer_groups_text(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     compute_metrics=compute_metrics
# )



# trainer.train()







