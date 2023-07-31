import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, Trainer, TrainingArguments, get_linear_schedule_with_warmup
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
import torch.nn.functional as F


class BertForSequenceClassificationText(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        dim = 512

        self.bert = BertModel(config).from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         self.classifier = torch.nn.Sequential(*[
# #                     torch.nn.Dropout(config.hidden_dropout_prob),
#                     torch.nn.Linear(in_features=self.config.hidden_size, out_features=dim),
#                     torch.nn.ReLU(),
#                     torch.nn.Dropout(config.hidden_dropout_prob),
#                     torch.nn.Linear(in_features=dim, out_features= config.num_labels)
#                 ])
        self.classifier = torch.nn.Sequential(
            # Original layers
            # torch.nn.Dropout(config.hidden_dropout_prob),
            torch.nn.Linear(in_features=self.config.hidden_size, out_features=dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.hidden_dropout_prob),

            torch.nn.Linear(in_features=dim, out_features=dim),  

            torch.nn.ReLU(),
            torch.nn.Dropout(config.hidden_dropout_prob),
            torch.nn.Linear(in_features=dim, out_features=config.num_labels)
        )


        self.init_weights()

    def forward(
        self,
#         annotator_ids = None, #YARA REMOVE THIS IF NOT TRAINER
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        mode = None,
        freeze = False
    ):

#         self.bert.requires_grad = False
        
        if freeze:
            #freeze bert params
            for param in self.bert.parameters():
#                 print("param", param)
                param.requires_grad = False

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

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
    
    




class BertForSequenceClassificationWithGroups(BertPreTrainedModel):
    def __init__(self, config): 
        super().__init__(config)
        self.num_labels = config.num_labels
        dim = 512

        self.bert = BertModel(config).from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size + config.group_embedding_dim, config.num_labels)
        self.classifier = torch.nn.Sequential(*[
#                     torch.nn.Dropout(config.hidden_dropout_prob),
                    torch.nn.Linear(in_features=self.config.hidden_size + config.group_embedding_dim, out_features=dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(config.hidden_dropout_prob),
                    torch.nn.Linear(in_features=dim, out_features= config.num_labels)
                ])
        variance = 0.01  
        std_deviation = torch.sqrt(torch.tensor(variance))

        self.group_embeddings = nn.Parameter(std_deviation*torch.randn(config.num_groups, config.group_embedding_dim))
        self.group_assignment = nn.Parameter(std_deviation*torch.randn(config.num_annotators, config.num_groups))
#         self.group_assignment = nn.Parameter(torch.empty(config.num_annotators, config.num_groups))
        
#         with torch.no_grad():
#             self.group_assignment.fill_(1)
#             self.group_assignment += torch.randn_like(self.group_assignment) / 100
#             self.group_assignment.relu_()
#             self.group_assignment /= self.group_assignment.sum(dim=1, keepdims=True)

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
        mode = None,
        freeze = False 
    ):
#         rater_group_assignment = torch.log(self.group_assignment[annotator_ids])
        rater_group_assignment = self.group_assignment[annotator_ids].log_softmax(dim=1)
        if freeze:
            #freeze bert params
            for param in self.bert.parameters():
#                 print("param", param)
                param.requires_grad = False
                
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
        pooled_output = pooled_output.unsqueeze(1).repeat(1,self.group_embeddings.size(0),1)
        a = self.group_embeddings.unsqueeze(0).repeat(pooled_output.shape[0], 1, 1)

        pooled_output = torch.cat((pooled_output, a), dim=2)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, rater_group_assignment 
    
class BertForSequenceClassificationWithAnnotators(BertPreTrainedModel):
    def __init__(self, config): 
        super().__init__(config)
        self.num_labels = config.num_labels
        dim = 512
        

        self.bert = BertModel(config).from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size + config.annotator_embedding_dim, config.num_labels)
        self.classifier = torch.nn.Sequential(*[
        #                     torch.nn.Dropout(config.hidden_dropout_prob),
                    torch.nn.Linear(in_features=self.config.hidden_size + config.annotator_embedding_dim, out_features=dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(config.hidden_dropout_prob),
                    torch.nn.Linear(in_features=dim, out_features= config.num_labels)
                ])

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
        freeze = False
    ):

        if freeze:
            #freeze bert params
            for param in self.bert.parameters():
                param.requires_grad = False

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
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    

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
            'annotator_id': annotator_id,
            'label': annotation,
            'disagreement': disagreement
        }
    
def project_simplex(v, z=1):
    v_sorted, _ = torch.sort(v, dim=0, descending=True)
    cssv = torch.cumsum(v_sorted, dim=0) - z
    ind = torch.arange(1, 1 + len(v)).to(dtype=v.dtype)
    cond = v_sorted - cssv / ind > 0
    rho = ind.masked_select(cond)[-1]
    tau = cssv.masked_select(cond)[-1] / rho
    w = torch.clamp(v - tau, min=0)
    return w
    
def train(model, device, train_data_loader, mode, freeze):
    assigned_groups = {}
    entropy = {}
    # Training loop
    num_epochs = 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay = 0.01) #lr=5e-5,
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay = 0.01) #lr=5e-5,
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
#     warmup_steps = 1000
    warmup_steps = 500
    
    # Warm-up scheduler
#     warmup_steps = min(warmup_steps, len(train_data_loader) * num_epochs)  # Cap warmup_steps based on the total number of steps
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda steps: min((steps+1)/warmup_steps, 1))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,             num_training_steps=len(train_data_loader)*num_epochs)


    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            annotator_ids = batch['annotator_id'].to(device)
            labels = batch['label'].to(device)
            if mode=="groups" :
                w, log_p = model(annotator_ids = annotator_ids, input_ids =input_ids, attention_mask = attention_mask, labels =                   labels, freeze = freeze)
                loss = torch.zeros(input_ids.size(0))
                for i in range(input_ids.size(0)):
                    loss[i] = - (w[i].log_softmax(dim=1) + log_p[i].reshape(-1, 1)).logsumexp(dim=0)[labels[i]]
                # Backward pass and optimization
                loss = torch.mean(loss)
            elif mode=="annotators":
                # Forward pass
                outputs = model(annotator_ids = annotator_ids, input_ids =input_ids, attention_mask = attention_mask, labels = labels, freeze = freeze)
                loss = outputs[0]
            else:
                outputs = model(input_ids =input_ids, attention_mask = attention_mask, labels = labels, freeze = freeze)
                loss = outputs[0]
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
#             if mode=="groups" :
#                 with torch.no_grad():
#                     for a in range(model.group_assignment.shape[0]):
#                         model.group_assignment[a] = project_simplex(model.group_assignment[a].cpu())

            total_loss += loss.item()
            
            scheduler.step()

        average_loss = total_loss / len(train_data_loader)

        print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {average_loss:.4f}')
#         get_accuracy(model, device, train_data_loader,assigned_groups, entropy, mode="groups")
        
class CustomTrainer_annotators_text(Trainer):
    def compute_loss(self, model, inputs, device=torch.device("cuda"), return_outputs=False):
        
        input_ids = inputs.get("input_ids").to(device)
        attention_mask = inputs.get("attention_mask").to(device)
        annotator_ids = inputs.get("annotator_id").to(device)
        labels = inputs.get("labels").to(device)
        
        outputs = model(annotator_ids = annotator_ids, input_ids =input_ids, attention_mask = attention_mask, labels =                     labels, freeze = False)
        loss = outputs[0]
        
        if return_outputs:
            return loss, outputs
        return loss
    
class CustomTrainer_text(Trainer):
    def compute_loss(self, model, inputs, device=torch.device("cuda"), return_outputs=False):
        
        input_ids = inputs.get("input_ids").to(device)
        attention_mask = inputs.get("attention_mask").to(device)
#         labels =  F.one_hot(inputs.get("labels").to(device).long(), num_classes=5)
        labels = inputs.get("labels").to(device)
        
#         labels = F.one_hot(labels.long(), num_classes=5)
        outputs = model(input_ids =input_ids, attention_mask = attention_mask, labels = labels)
        loss = outputs[0]
        
        if return_outputs:
            return loss, outputs
        return loss
    
class CustomTrainer_groups_text(Trainer):
    def compute_loss(self, model, inputs, device=torch.device("cuda"), return_outputs=False):
        

        input_ids = inputs.get("input_ids").to(device)
        attention_mask = inputs.get("attention_mask").to(device)
        annotator_ids = inputs.get("annotator_id").to(device)
        labels = inputs.get("labels").to(device)
        

        w, log_p = model(annotator_ids = annotator_ids, input_ids =input_ids, attention_mask = attention_mask, labels = labels, freeze = False)
        loss = torch.zeros(input_ids.size(0))
        for i in range(input_ids.size(0)):
            loss[i] = - (w[i].log_softmax(dim=1) + log_p[i].reshape(-1, 1)).logsumexp(dim=0)[labels[i].long()]
        # Backward pass and optimization
        loss = torch.mean(loss)

        
        if return_outputs:
            return loss, w
        return loss

    


# class CustomDataCollator:
#     def __call__(self, examples):
#         labels = [example['label'] for example in examples]
#         input_ids = [example['input_ids'] for example in examples]
#         attention_mask = [example['attention_mask'] for example in examples]
#         annotator_id = [example['annotator_id'] for example in examples]
#         disagreement = [example['disagreement'] for example in examples]

#         batch = {
#             'label': torch.tensor(labels),
#             'input_ids': torch.stack(input_ids),
#             'attention_mask': torch.stack(attention_mask),
#             'annotator_id': torch.tensor(annotator_id),
#             'disagreement': torch.tensor(disagreement),
#         }

#         return batch

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
      'accuracy': acc,
    }


        
def train_trainer(model, device, train_data_loader, test_data_loader, mode, freeze):
    
    # Define the training arguments
#     training_args = TrainingArguments(
#         output_dir='./results',           # Directory to save checkpoints and logs
#         num_train_epochs=10,              # Number of training epochs
#         per_device_train_batch_size=16,    # Batch size per device during training
#         learning_rate=1e-3,               # Learning rate
#         weight_decay=0.1,                 # Weight decay
#         logging_dir='./logs',             # Directory for storing logs
#         logging_steps=100,                # Limit on the total number of checkpoints to save
#         save_strategy="no",                   # Number of steps before saving a checkpoint
#         warmup_steps=500,                 # Number of warmup steps for learning rate schedule
#         optim = "adamw_torch",
#         remove_unused_columns=False
#     )
    training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=250,
    evaluation_strategy = "epoch",
    logging_strategy="epoch",
    remove_unused_columns=False
    )

    #set training args device to mps
    training_args._n_gpu = 2
    training_args._device = device
    
    
#     custom_data_collator = CustomDataCollator()
    if mode=="groups":
        trainer = CustomTrainer_groups_text(
            model = model,
            args=training_args,
            train_dataset=train_data_loader.dataset,
#             data_collator = custom_data_collator
        )
            
    elif mode=="annotators":
        
        trainer = CustomTrainer_annotators_text(
        model=model,
        args=training_args,
        train_dataset=train_data_loader.dataset,
#         data_collator=custom_data_collator,
        )
    else:
        
        
        # Define the training arguments
#         training_args = TrainingArguments(
#             output_dir='./results',           # Directory to save checkpoints and logs
#             num_train_epochs=10,              # Number of training epochs
#             per_device_train_batch_size=16,    # Batch size per device during training
#             learning_rate=1e-3,               # Learning rate
#             weight_decay=0.1,                 # Weight decay
#             logging_dir='./logs',             # Directory for storing logs
#             logging_steps=100,  
#             save_strategy="no",
#             warmup_steps=500,                 # Number of warmup steps for learning rate schedule
#             optim = "adamw_torch",
#         )
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=250,
            evaluation_strategy = "epoch",
            logging_strategy="epoch",
            remove_unused_columns=False
            )
        

        #set training args device to mps
        training_args._n_gpu = 1
        training_args._device = device
        
        trainer = CustomTrainer_text(
        model=model,
        args=training_args,
        train_dataset=train_data_loader.dataset,
        eval_dataset = test_data_loader.dataset,
        compute_metrics=compute_metrics
#         data_collator=custom_data_collator,
        )

    # Train the model
    trainer.train() 
    
#     trainer.save(model)
    
    model.eval()
    
    
    predictions_train, labels_train, metrics_train = trainer.predict(train_data_loader.dataset)
    predicted_labels_train = predictions_train.argmax(axis=1)
    accuracy_train = accuracy_score(labels_train, predicted_labels_train)
    print(f"Train Accuracy: {accuracy_train}")
    
    predictions_test, labels_test, metrics_test = trainer.predict(test_data_loader.dataset)
    predicted_labels_test = predictions_test.argmax(axis=1)
    accuracy_test = accuracy_score(labels_test, predicted_labels_test)
    print(f"Test Accuracy: {accuracy_test}")


def get_accuracy(model, device, data_loader, assigned_groups, entropy, mode):
    model.eval()
    correct_predictions_disagreement = 0
    total_predictions_disagreement = 0
    correct_predictions_agreement = 0
    total_predictions_agreement = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            annotator_ids = batch['annotator_id'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            disagreement = batch['disagreement'].to(device)

            # Forward pass
            if mode=="groups":
                w, log_p = model(annotator_ids, input_ids, attention_mask=attention_mask)
                p = torch.exp(log_p)
                entropy_ar = - torch.sum(p*log_p, axis=1)
                best_group = log_p.argmax(dim=1)
                
                
                for annotator_id, ent in zip(annotator_ids, entropy_ar):
                    ent = float(ent.item())
                    annotator_id = int(annotator_id.item())
                    if annotator_id in entropy:
                #             assigned_groups[annotator_id].append(bg)
                        entropy[annotator_id] = list(set(entropy[annotator_id] + [ent]))
                    else:
                        entropy[annotator_id] = [ent]

                for annotator_id, bg in zip(annotator_ids, best_group):
                    bg = int(bg.item())
                    annotator_id = int(annotator_id.item())
                    if annotator_id in assigned_groups:
            #             assigned_groups[annotator_id].append(bg)
                        assigned_groups[annotator_id] = list(set(assigned_groups[annotator_id] + [bg]))
                    else:
                        assigned_groups[annotator_id] = [bg]
                
                
                w = w[range(len(w)), best_group]
                _, predicted = torch.max(w, 1)

                
            elif mode == "annotators":
                outputs = model(annotator_ids, input_ids, attention_mask=attention_mask)
                logits = outputs[0]

                # Get predicted labels
                _, predicted = torch.max(logits, dim=1)
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs[0]

                # Get predicted labels
                _, predicted = torch.max(logits, dim=1)
                
            labels_agreement = labels[~disagreement]
            labels_disagreement = labels[disagreement]
            predicted_agreement = predicted[~disagreement]
            predicted_disagreement = predicted[disagreement]        

            correct_predictions_disagreement += (predicted_disagreement == labels_disagreement).sum().item()
            total_predictions_disagreement += labels_disagreement.size(0)
            correct_predictions_agreement += (predicted_agreement == labels_agreement).sum().item()
            total_predictions_agreement += labels_agreement.size(0)

            correct_predictions += (predicted == labels).sum().item()
            total_predictions+= labels.size(0)
    

    accuracy_disagreement = correct_predictions_disagreement / total_predictions_disagreement
    accuracy_agreement = correct_predictions_agreement / total_predictions_agreement
    accuracy = correct_predictions / total_predictions
    
    print("final dict assigned_groups:",len(assigned_groups), assigned_groups)
    print("final dict entropy:",len(entropy), entropy)
    print("accuracy", accuracy)
    print("accuracy_disagreement", accuracy_disagreement)
    print("accuracy_agreement", accuracy_agreement)
    
    
    
    return accuracy, accuracy_disagreement, accuracy_agreement
#     return accuracy
