
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
from scipy.special import logsumexp
from sklearn.model_selection import GroupShuffleSplit 
from models import utils
from models import mace, latent_annotator_clustering
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score



train_df = pd.read_csv('../data/toxicity_ratings/train.csv',delimiter=',', encoding='latin-1')
test_df = pd.read_csv('../data/toxicity_ratings/test.csv', delimiter=',')
annotators_df = pd.read_csv('../data/toxicity_ratings/annotators.csv', delimiter=',')
df = pd.concat([test_df, train_df])

total_annotator_ids = train_df['annotator_id'].unique().tolist()




pivot_df = pd.pivot_table(
    train_df,
    values='annotation',
    index='sentence_id',
    columns='annotator_id', 
    # aggfunc=lambda x: ''.join(x) # Concatenate multiple annotations if they exist for a single unit/annotator pair
)

pivot_df = pivot_df.fillna(-1)




pivot_df.index.name = None
pivot_df.columns.name = None



data = list()

# Iterate over the rows of the DataFrame
for index, row in pivot_df.iterrows():
    data.append(list())
    for a in row.index:
    # Extract the annotator and annotation values from the row
        annotator = a
        annotation = int(row[a])
        # print(annotator, annotation)
        if(annotation  != -1):
        # Append the (annotator, annotation) pair to the data list
            data[-1].append((annotator, annotation))






A_ls, B_ls, C_ls, elbos_ls = latent_annotator_clustering.evaluate(data, len(df["annotation"].unique()), len(df["annotator_id"].unique()), 15,100, smoothing=True, logspace=True)




clusters = B_ls.argmax(axis=1)



# how many annotators are in each cluster
np.unique(clusters, return_counts=True)



# order anxiety_score_df and demographics_df by respondent_id
annotators_df = annotators_df.sort_values(by=['annotator_id'])



df_indices = train_df['annotator_id'].unique().tolist()  # Get the indices present in the DataFrame column
clusters = [item for i,item in enumerate(clusters) if i in df_indices] 




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


print(np.bincount(clusters))

