import numpy as np
from torchtext.vocab import FastText, vocab
import torch

def softplus_np(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


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