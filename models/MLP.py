import pandas as pd
import numpy as np
import torch

import torch.nn as nn
from torch import optim

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, custom_weights=None, pretrained_embeddings=None, text_only=False):
        super(Classifier, self).__init__()
        self.rater_embedding = nn.Embedding(input_dim['id_vocab_size'], input_dim['id_embedding_dim'],_weight=custom_weights)
        self.text_embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        self.text_only = text_only

        if text_only:
            self.fc1 = nn.Linear(input_dim["text_embedding_dim"], hidden_dim)
        else:
            self.fc1 = nn.Linear(input_dim["text_embedding_dim"]+input_dim["id_embedding_dim"], hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, token_ids, rater_id):
        token_embeddings = self.text_embedding(token_ids) 
        sentence_embedding = token_embeddings.mean(dim=1) #/len(token_ids)  
        rater_emb = self.rater_embedding(rater_id)
        
        if (self.text_only):
            x = sentence_embedding
        else:
            x = torch.cat((sentence_embedding, rater_emb), dim=1)
        # x = sentence_embedding
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ClassifierWithGroups(nn.Module):
    def __init__(self, num_groups, input_dim, hidden_dim, output_dim, custom_weights=None, pretrained_embeddings=None):
        super(ClassifierWithGroups, self).__init__()

        self.group_embeddings = nn.Parameter(torch.randn(num_groups, input_dim['id_embedding_dim']))
        self.group_assignment = nn.Parameter(torch.randn(input_dim['id_vocab_size'], num_groups))

        if pretrained_embeddings is None:
            self.text_embedding = nn.Embedding(input_dim["num_words"], input_dim["text_embedding_dim"])
        else:
            self.text_embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)

        self.fc1_part_a = nn.Linear(input_dim["text_embedding_dim"], hidden_dim)
        self.fc1_part_b = nn.Linear(input_dim["id_embedding_dim"], hidden_dim)
        self.fc1 = nn.Linear(input_dim["text_embedding_dim"]+input_dim["id_embedding_dim"], hidden_dim)

        
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim
        
    def forward(self, token_ids, rater_id):

        rater_group_assignment = self.group_assignment[rater_id].log_softmax(dim=1)


        token_embeddings = self.text_embedding(token_ids) 
        sentence_embedding = token_embeddings.mean(dim=1)
       
        a = self.fc1_part_a(sentence_embedding)
        b = self.fc1_part_b(self.group_embeddings)

        b_expanded = b.unsqueeze(0).expand(a.size(0), -1, -1)

        # Perform element-wise addition
        result = a.unsqueeze(1) + b_expanded
        h = self.relu(result)
        w = self.fc2(h)
        return w, rater_group_assignment

def train(pretrained_embeddings, text_ids, annotator_ids, annotations, nb_classes, mode, nb_groups=0):

    X_text = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in text_ids], batch_first=True)
    X_rater_id = torch.tensor(annotator_ids.values)

    y = torch.tensor(annotations.values)

    input_dim = {
            "id_embedding_dim": 100,
            "id_vocab_size": len(annotator_ids.unique()),
            "text_embedding_dim": 300,
            "num_words": X_text.shape[1]
        }
    hidden_dim = 50 # dimension of the hidden layer
    output_dim = nb_classes # number of classes

    #CASE NORMAL INIT
    custom_weights = torch.randn(input_dim['id_vocab_size'], input_dim['id_embedding_dim'], requires_grad=True)
    # CASE UNIFORM INIT
    # custom_weights = 2*(torch.rand(input_dim['id_vocab_size'], input_dim['id_embedding_dim'], requires_grad=True))-1
    print(torch.mean(custom_weights),torch.var(custom_weights))

    if mode == "text":
        classifier = Classifier(input_dim, hidden_dim, output_dim, custom_weights=custom_weights, pretrained_embeddings=pretrained_embeddings, text_only=True)
    elif mode == "text_annotators":
        classifier = Classifier(input_dim, hidden_dim, output_dim, custom_weights=custom_weights, pretrained_embeddings=pretrained_embeddings, text_only=False)
    elif mode == "text_groups":
        classifier = ClassifierWithGroups(nb_groups, input_dim, hidden_dim, output_dim, custom_weights=custom_weights, pretrained_embeddings=pretrained_embeddings)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), eps=1e-10, lr=0.001)#, weight_decay=0.0001)

    num_epochs = 20
    batch_size = 32
    for epoch in range(num_epochs):
        for i in range(0, len(X_text), batch_size):
            # get the batch of sentence embeddings and labels
            batch_X_text = X_text[i:i+batch_size]
            batch_X_id = X_rater_id[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            # zero the gradients
            optimizer.zero_grad()
            # forward pass
            if (mode == "text" or mode == "text_annotators"):
                outputs = classifier(batch_X_text, batch_X_id)
                loss = criterion(outputs, batch_y)
            elif mode == "text_groups":
                w, log_p = classifier(batch_X_text, batch_X_id)
                loss = torch.zeros(batch_size)
                for i in range(batch_X_text.shape[0]):
                    loss[i] = - (w[i].log_softmax(dim=1) + log_p[i].reshape(-1, 1)).logsumexp(dim=0)[batch_y[i]]

            # w_log_softmax = w.log_softmax(dim=2)
            # log_p_reshaped = log_p.unsqueeze(2)
            # log_sum_exp = (w_log_softmax + log_p_reshaped).logsumexp(dim=1)
            # loss = -log_sum_exp[batch_y]

            loss = torch.mean(loss)
            loss.backward()

            optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    return classifier

def test(text_ids, annotator_ids, annotations, classifier, mode):
    X_test_text = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in text_ids], batch_first=True)
    X_test_id = torch.tensor(annotator_ids.values)

    y_test = torch.tensor(annotations.values)

    with torch.no_grad():
        if mode == "text" or mode == "text_annotators":
            outputs = classifier(X_test_text, X_test_id)
            _, predicted = torch.max(outputs.data, 1)
        elif mode == "text_groups":
            w, log_p = classifier(X_test_text, X_test_id)
            best_group = log_p.argmax(dim=1)
            w = w[range(len(w)), best_group]
            _, predicted = torch.max(w, 1)
        correct = (predicted == y_test).sum().item()
        print('Accuracy of the model on the test set: {} %'.format(100 * correct / len(y_test)))
    
    return predicted