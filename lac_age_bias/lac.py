
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



train_df = pd.read_csv('../data/age_bias/train_new_agr.csv',delimiter=',', encoding='latin-1')
test_df = pd.read_csv('../data/age_bias/test_new_agr.csv', delimiter=',')
demographics_df = pd.read_csv('../data/age_bias/demographics.csv', delimiter=',')
anxiety_score_df = pd.read_csv('../data/age_bias/anxiety_score.csv', delimiter=',')
df = pd.concat([test_df, train_df])

total_annotator_ids = train_df['annotator_id'].unique().tolist()




pivot_df = pd.pivot_table(
    train_df,
    values='annotation',
    index='unit_id',
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
anxiety_score_df = anxiety_score_df.sort_values(by=['respondent_id'])
demographics_df = demographics_df.sort_values(by=['respondent_id'])



df_indices = train_df['annotator_id'].unique().tolist()  # Get the indices present in the DataFrame column
clusters = [item for i,item in enumerate(clusters) if i in df_indices] 


bins = [20, 30, 40, 50, 60, 70, 80, 90, 100]  # These are the bin edges

# Use pd.cut() to categorize the anxiety scores
anxiety_score_df["anxiety_level"] = pd.cut(anxiety_score_df["anxiety_score"], bins=bins)

# Display the updated DataFrame
print(anxiety_score_df)


data = pd.DataFrame({'Cluster': clusters, 'Anxiety Level': anxiety_score_df["anxiety_level"]})
grouped_data = data.groupby(['Cluster', 'Anxiety Level']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Anxiety Level', values='Count')
pivot_table.plot(kind='bar', stacked=True)

# Save the plot as an image file
plt.savefig('plots/anxiety_level.png')

# Close the plot to free up memory
plt.close()




data = pd.DataFrame({'Cluster': clusters, 'Age Group': demographics_df.iloc[:, 1]})
grouped_data = data.groupby(['Cluster', 'Age Group']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Age Group', values='Count')
pivot_table.plot(kind='bar', stacked=True)

# Save the plot as an image file
plt.savefig('plots/age_group.png')

# Close the plot to free up memory
plt.close()


data = pd.DataFrame({'Cluster': clusters, 'Race Group': demographics_df.iloc[:, 2]})
grouped_data = data.groupby(['Cluster', 'Race Group']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Race Group', values='Count')
pivot_table.plot(kind='bar', stacked=True)

# Save the plot as an image file
plt.savefig('plots/race_group.png')

# Close the plot to free up memory
plt.close()


data = pd.DataFrame({'Cluster': clusters, 'Hespanic/Latino': demographics_df.iloc[:, 4]})
grouped_data = data.groupby(['Cluster', 'Hespanic/Latino']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Hespanic/Latino', values='Count')
pivot_table.plot(kind='bar', stacked=True)

# Save the plot as an image file
plt.savefig('plots/hespanic_latino.png')

# Close the plot to free up memory
plt.close()


data = pd.DataFrame({'Cluster': clusters, 'Area where raised': demographics_df.iloc[:, 5]})
grouped_data = data.groupby(['Cluster', 'Area where raised']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Area where raised', values='Count')
pivot_table.plot(kind='bar', stacked=True)

# Save the plot as an image file
plt.savefig('plots/area_raised.png')

# Close the plot to free up memory
plt.close()


data = pd.DataFrame({'Cluster': clusters, 'Current Area': demographics_df.iloc[:, 6]})
grouped_data = data.groupby(['Cluster', 'Current Area']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Current Area', values='Count')
pivot_table.plot(kind='bar', stacked=True)

# Save the plot as an image file
plt.savefig('plots/current_area.png')

# Close the plot to free up memory
plt.close()


data = pd.DataFrame({'Cluster': clusters, 'Current Region': demographics_df.iloc[:, 7]})
grouped_data = data.groupby(['Cluster', 'Current Region']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Current Region', values='Count')
pivot_table.plot(kind='bar', stacked=True)

# Save the plot as an image file
plt.savefig('plots/current_region.png')

# Close the plot to free up memory
plt.close()


data = pd.DataFrame({'Cluster': clusters, 'Annual Income': demographics_df.iloc[:, 8]})
grouped_data = data.groupby(['Cluster', 'Annual Income']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Annual Income', values='Count')
pivot_table.plot(kind='bar', stacked=True)

# Save the plot as an image file
plt.savefig('plots/annual_income.png')

# Close the plot to free up memory
plt.close()


data = pd.DataFrame({'Cluster': clusters, 'Education': demographics_df.iloc[:, 9]})
grouped_data = data.groupby(['Cluster', 'Education']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Education', values='Count')
pivot_table.plot(kind='bar', stacked=True)

# Save the plot as an image file
plt.savefig('plots/education.png')

# Close the plot to free up memory
plt.close()


data = pd.DataFrame({'Cluster': clusters, 'Employment': demographics_df.iloc[:, 10]})
grouped_data = data.groupby(['Cluster', 'Employment']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Employment', values='Count')
pivot_table.plot(kind='bar', stacked=True)
# Save the plot as an image file
plt.savefig('plots/employment.png')

# Close the plot to free up memory
plt.close()

data = pd.DataFrame({'Cluster': clusters, 'Living Situation': demographics_df.iloc[:, 11]})
grouped_data = data.groupby(['Cluster', 'Living Situation']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Living Situation', values='Count')
pivot_table.plot(kind='bar', stacked=True)
# Save the plot as an image file
plt.savefig('plots/living_situation.png')

# Close the plot to free up memory
plt.close()


data = pd.DataFrame({'Cluster': clusters, 'Political Identification': demographics_df.iloc[:, 13]})
grouped_data = data.groupby(['Cluster', 'Political Identification']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Political Identification', values='Count')
pivot_table.plot(kind='bar', stacked=True)
# Save the plot as an image file
plt.savefig('plots/political_identification.png')

# Close the plot to free up memory
plt.close()

data = pd.DataFrame({'Cluster': clusters, 'Gender': demographics_df.iloc[:, 14]})
grouped_data = data.groupby(['Cluster', 'Gender']).size().reset_index(name='Count')
pivot_table = grouped_data.pivot(index='Cluster', columns='Gender', values='Count')
pivot_table.plot(kind='bar', stacked=True)

plt.savefig('plots/gender.png')

# Close the plot to free up memory
plt.close()



print(np.bincount(clusters))

