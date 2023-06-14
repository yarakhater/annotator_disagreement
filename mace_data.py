import pandas as pd

import numpy as np
from scipy.special import logsumexp
from sklearn.model_selection import GroupShuffleSplit 
from models import utils
from models import mace 

data_files = ["./data/temp.standardized.tsv", "./data/wsd.standardized.tsv", "./data/rte.standardized.tsv"]

datasets = ["Temporal", "WSD", "RTE"]

for d, data_file in enumerate(data_files):

    df = pd.read_csv(data_file, delimiter='\t')

    pivot_df = pd.pivot_table(
    df,
    values='response',
    index='orig_id',
    columns='!amt_worker_ids'
    )

    pivot_df = pivot_df.fillna(-1)
    pivot_df.index.name = None
    pivot_df.columns.name = None

    #dataframe of original ids and their corresponding gold labels
    gold_df = df[["orig_id", "gold"]].drop_duplicates()

    # dictionary of annotator ids and integer indices
    annotator_ids = df["!amt_worker_ids"].unique()
    annotator_id_to_index = {annotator_id: i for i, annotator_id in enumerate(annotator_ids)}
    # replace annotator ids with integer indices
    pivot_df.columns = [annotator_id_to_index[annotator_id] for annotator_id in pivot_df.columns]
    # dictionary of original ids and integer indices
    orig_ids = df["orig_id"].unique()
    orig_id_to_index = {orig_id: i for i, orig_id in enumerate(orig_ids)}
    # replace original ids with integer indices
    pivot_df.index = [orig_id_to_index[orig_id] for orig_id in pivot_df.index]
    # order annotators by their indices and original ids by their indices
    pivot_df = pivot_df.sort_index(axis=0).sort_index(axis=1)

    data = list()

    for index, row in pivot_df.iterrows():
        data.append(list())
        for a in row.index:
            annotator = a
            annotation = int(row[a])
            if(annotation  != -1 ): 
                data[-1].append((annotator, annotation - gold_df["gold"].min())) # annotation-1 for wsd and tmp and annotation for rte
    
    # run MACE
    A_ls, B_ls, T_ls, elbos_ls = mace.evaluate(data, len(df["response"].unique()), len(df["!amt_worker_ids"].unique()), 100, smoothing=True, logspace=True)

    # replace original ids with integer indices
    gold_df.index = [orig_id_to_index[orig_id] for orig_id in gold_df["orig_id"]]
    # sort gold_df by the indices
    gold_df = gold_df.sort_index(axis=0)

    gold_A = np.array(gold_df["gold"] - gold_df["gold"].min() )  # gold_df["gold"]-1 for wsd and tmp and gold_df["gold"] for rte

    accuracy = np.mean(gold_A  == np.argmax(A_ls, axis=1)) 

    # percentage of annotating the correct label for each annotator
    annotator_accuracy = np.zeros(len(annotator_ids))
    for i, annotator_id in enumerate(annotator_ids):
        annotator_df = df[df["!amt_worker_ids"] == annotator_id]
        annotator_accuracy[i] = np.mean(annotator_df["response"] == annotator_df["gold"])
    
    # pearson correlation between annotator accuracy and T
    correlation = np.corrcoef(annotator_accuracy, T_ls)

    print("Dataset: {}".format(datasets[d]))
    print("Accuracy: {:.2f}".format(accuracy))
    print("Pearson correlation: {:.2f}".format(correlation[0, 1]))


