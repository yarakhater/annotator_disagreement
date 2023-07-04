import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from torch.utils.data import Dataset



class BertForSequenceClassificationText(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config).from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
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
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
    
    




class BertForSequenceClassificationWithGroups(BertPreTrainedModel):
    def __init__(self, config): 
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config).from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size + config.group_embedding_dim, config.num_labels)
        variance = 0.01  
        std_deviation = torch.sqrt(torch.tensor(variance))

        self.group_embeddings = nn.Parameter(std_deviation*torch.randn(config.num_groups, config.group_embedding_dim))
        self.group_assignment = nn.Parameter(std_deviation*torch.randn(config.num_annotators, config.num_groups))

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
        
        rater_group_assignment = self.group_assignment[annotator_ids].log_softmax(dim=1)
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

        self.bert = BertModel(config).from_pretrained('bert-base-uncased')
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
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
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

def train(model, device, train_data_loader, mode):
    # Training loop
    num_epochs = 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay = 0.1)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)


    for epoch in range(num_epochs):
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

        print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {average_loss:.4f}')

def test(model, device, test_data_loader, mode):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in test_data_loader:
            input_ids = batch['input_ids'].to(device)
            annotator_ids = batch['annotator_id'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            if mode=="groups":
                w, log_p = model(annotator_ids, input_ids, attention_mask=attention_mask)
                best_group = log_p.argmax(dim=1)
                w = w[range(len(w)), best_group]
                _, predicted = torch.max(w, 1)

                
            elif mode == "annotators":
                outputs = model(annotator_ids, input_ids, attention_mask=attention_mask)
                logits = outputs[0]

                # Get predicted labels
                _, predicted = torch.max(logits, dim=1)
            else:
                print("-else")
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs[0]

                # Get predicted labels
                _, predicted = torch.max(logits, dim=1)

            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    print(f'Accuracy: {accuracy:.4f}')

