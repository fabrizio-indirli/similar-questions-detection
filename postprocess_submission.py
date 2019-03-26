from collections import defaultdict
import numpy as np
import pandas as pd
import os
import networkx as nx

n_repeats=2
duplicate_threshold = 0.7
not_duplicate_threshold = 0.1
maximum_update = 0.2
upper_bound = 0.98
lower_bound = 0.01

print("Load data...")
train = pd.read_csv("./data/train.csv", names=['row_ID', 'text_a_ID', 'text_b_ID', 'text_a_text', 'text_b_text', 'have_same_meaning'], index_col=0)
test = pd.read_csv("./data/test.csv", names=['row_ID', 'text_a_ID', 'text_b_ID', 'text_a_text', 'text_b_text', 'have_same_meaning'], index_col=0)
submission_sample = pd.read_csv("./sample_submission_file.csv")

submission = pd.read_csv("./predictions/submission.csv")
test_predictions = submission.Score.values

for i in range(n_repeats):
    nodes = pd.concat([train.text_a_ID, train.text_b_ID, test.text_a_ID,test.text_b_ID]).values
    edges = pd.concat([train.loc[train.have_same_meaning==1, ["text_a_ID", "text_b_ID"]], test.loc[test_predictions > duplicate_threshold, ["text_a_ID", "text_b_ID"]]]).values

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    neighbor_count = test.apply(lambda row: len(list(nx.common_neighbors(g, row['text_a_ID'], row['text_b_ID']))), axis=1)
    count = 0
    for i, neigh_count in enumerate(neighbor_count):
        if neigh_count > 0 and test_predictions[i] < upper_bound:
            test_predictions[i] += min(maximum_update, (upper_bound - test_predictions[i]) / 2)     
            count +=1       
    print("Updated {} predictions".format(count))

for i in range(n_repeats):
    nodes = pd.concat([train.text_a_ID, train.text_b_ID, test.text_a_ID,test.text_b_ID]).values
    edges = pd.concat([train.loc[train.have_same_meaning==0, ["text_a_ID", "text_b_ID"]], test.loc[test_predictions < not_duplicate_threshold, ["text_a_ID", "text_b_ID"]]]).values

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    neighbor_count = test.apply(lambda row: len(list(nx.common_neighbors(g, row['text_a_ID'], row['text_b_ID']))), axis=1)
    count = 0
    for i, neigh_count in enumerate(neighbor_count):
        if neigh_count > 0 and test_predictions[i] > lower_bound:
            test_predictions[i] -= min(maximum_update, (test_predictions[i] - lower_bound) / 2)
            count +=1       
    print("Updated {} predictions".format(count))

submission = pd.DataFrame({"Id":submission.Id, "Score":test_predictions})
submission.to_csv("predictions/postprocessed_submission.csv", index=False)

