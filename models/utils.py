import numpy as np
from torchtext.vocab import FastText, vocab
from torch.utils.data import Dataset
import torch

def softplus_np(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

# for mlp model
def clean_text(df, column_name, stop_words):

    # Remove "NEWLINE_TOKEN"
    df[column_name] = df[column_name].str.replace('NEWLINE_TOKEN', '')

    # Remove punctuation
    df[column_name] = df[column_name].str.replace('[\W_]', ' ')

    # Convert to lowercase
    df[column_name] = df[column_name].str.lower()

    # Remove stopwords
    df[column_name] = df[column_name].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Remove numbers
    df[column_name] = df[column_name].str.replace('\d+', '')

    return df

def get_fasttext():
    unk_token = "<unk>"
    unk_index = 1
    pad_token = "<pad>"
    pad_index = 0

    fasttext_vectors = FastText()
    fasttext_vocab = vocab(fasttext_vectors.stoi)
    fasttext_vocab.insert_token(unk_token,unk_index)
    fasttext_vocab.insert_token(pad_token,pad_index)

    fasttext_vocab.set_default_index(unk_index)
    pretrained_embeddings = fasttext_vectors.vectors
    pretrained_embeddings = torch.cat((torch.zeros(2,pretrained_embeddings.shape[1]),pretrained_embeddings))

    return pretrained_embeddings, fasttext_vocab

# for bert model
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, labels):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = self.data.iloc[index]['comment']
        annotator_id = self.data.iloc[index]['worker_id']
        annotation = self.data.iloc[index]['toxic_score']

        # Tokenize the sentence and annotator_id
        inputs = self.tokenizer.encode_plus(
            sentence,
            # annotator_id,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        annotator_id = torch.tensor(annotator_id, dtype=torch.long)
        # annotator_embedding = torch.tensor(int(annotator_id), dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'annotator_id': annotator_id,
            # 'annotator_embedding': annotator_embedding,  
            'label': annotation
        }