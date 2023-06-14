import pandas as pd

import numpy as np
from scipy.special import logsumexp
from sklearn.model_selection import GroupShuffleSplit 
from models import utils
from models import mace, latent_annotator_clustering
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

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

    # replace original ids with integer indices
    gold_df.index = [orig_id_to_index[orig_id] for orig_id in gold_df["orig_id"]]
    # sort gold_df by the indices
    gold_df = gold_df.sort_index(axis=0)

    gold_A = np.array(gold_df["gold"] - gold_df["gold"].min() )  # gold_df["gold"]-1 for wsd and tmp and gold_df["gold"] for rte


    A_ls, B_ls, C_ls, elbos_ls = latent_annotator_clustering.evaluate(data, len(df["response"].unique()), len(df["!amt_worker_ids"].unique()), 2, num_iters = 100, num_restarts=10, logspace = True, smoothing = False)


    print(gold_A, A_ls.argmax(axis=1))
    print(100 * homogeneity_score(gold_A, A_ls.argmax(axis=1)))
    print(100 * completeness_score(gold_A, A_ls.argmax(axis=1)))
    print(100 * v_measure_score(gold_A, A_ls.argmax(axis=1)))
    print("Dataset: {}".format(datasets[d]))
    print("Gold:", gold_A)
    print("Predicted:", A_ls.argmax(axis=1))


