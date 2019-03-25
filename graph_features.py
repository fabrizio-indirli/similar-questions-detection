#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import re
import networkx as nx

# Load data
train = pd.read_csv("./train.csv", names=['row_ID', 'text_a_ID', 'text_b_ID', 'text_a_text', 'text_b_text', 'have_same_meaning'], index_col=0)
test = pd.read_csv("./test.csv", names=['row_ID', 'text_a_ID', 'text_b_ID', 'text_a_text', 'text_b_text', 'have_same_meaning'], index_col=0)
submission_sample = pd.read_csv("./sample_submission_file.csv")

train_distance_features = pd.read_csv("data/distance_features_train.csv")
test_distance_features = pd.read_csv("data/distance_features_test.csv")

weight_col = "tfidf_dist_cosine"

train[weight_col] = train_distance_features[weight_col]
test[weight_col] = test_distance_features[weight_col]

# Hyperparameters
max_freq = 50
max_neighbors = 30
n_k_cores = 11
max_level = 3

# Start computation
all_question_ids = pd.concat([train["text_a_ID"], train["text_b_ID"], test["text_a_ID"], test["text_b_ID"]])
unique_question_ids = all_question_ids.unique()

def shortestPathShortness(row):
    g.remove_edge(row['text_a_ID'], row['text_b_ID'])
    try:
        length = nx.shortest_path_length(g, row['text_a_ID'], row['text_b_ID'], weight="weight")
        if length != 0:
            sps = 1 / length
        else:
            sps = 0
    except nx.NetworkXNoPath:
        sps=0
    g.add_edge(row['text_a_ID'], row['text_b_ID'], weight=row[weight_col])
    return sps

def get_neighbors(qid):
    neighbors = nx.single_source_shortest_path_length(g, qid, cutoff=2)
    neighbors_df = pd.DataFrame(list(zip(neighbors.keys(), neighbors.values())), index=neighbors.keys(), columns=["qid", "n_level"])

    neighbors = []
    for i in range(1, max_level+1):
        neighbors.append(neighbors_df[neighbors_df.n_level==i].qid.values)
    return neighbors


print("Build Graph...")
nodes = pd.concat([train.text_a_ID, train.text_b_ID, test.text_a_ID,test.text_b_ID]).values
edges = pd.concat([train[["text_a_ID", "text_b_ID", weight_col]], test[["text_a_ID", "text_b_ID", weight_col]]]).values

g = nx.Graph()
g.add_nodes_from(nodes)
#g.add_edges_from(edges)
for e in edges:
    g.add_edge(int(e[0]), int(e[1]), weight=e[2])
g.remove_edges_from(g.selfloop_edges())

print("Compute question specific features...")
df_questions = pd.DataFrame(unique_question_ids, columns=["qid"])
df_questions.index = df_questions.qid

print("--> Compute k cores...")
df_questions["k_core"] = 0
for i in range(2,n_k_cores):
    print("\t--> core {}".format(i))
    k_core = nx.k_core(g, k=i).nodes()
    df_questions.loc[df_questions.qid.isin(k_core), "k_core"] = i

print("--> Compute neighbors...")  
neighbors = df_questions.qid.apply(get_neighbors)
for i in range(1, max_level+1):
    df_questions["neighbors" + str(i)] = neighbors.apply(lambda x: set(x[i-1]))

print("--> Compute question frequency...")  
df_questions["frequency"] = all_question_ids.value_counts()

print("--> Compute page rank...") 
pageranks = nx.pagerank(g, weight='weight')
df_questions["page_rank"] = df_questions.qid.apply(lambda qid: pageranks[qid])

print("--> Compute closeness centrality...") 
closeness_centrality = nx.closeness_centrality(g)
df_questions["closeness_centrality"] = df_questions.qid.apply(lambda qid: closeness_centrality[qid])

print("--> Compute clustering...") 
clustering = nx.clustering(g, weight='weight')
df_questions["clustering"] = df_questions.qid.apply(lambda qid: clustering[qid])
 
print("--> Compute eigenvector centrality...") 
eigenvector_centrality = nx.eigenvector_centrality(g, weight='weight')
df_questions["eigenvector_centrality"] = df_questions.qid.apply(lambda qid: eigenvector_centrality[qid])


def preprocess(df):
    df_features = pd.DataFrame(index=df.index)
    df_intermediate = pd.DataFrame(index=df.index)
    
    print("--> Compute shortest path shortness...")
    df_features["shortest_path_shortness"] = df.apply(lambda x: shortestPathShortness(x), axis=1)
        
    print("--> Compute frequency features...")
    df_intermediate["freq_a"] = df_questions.loc[df.text_a_ID, "frequency"].values
    df_intermediate["freq_b"] = df_questions.loc[df.text_b_ID, "frequency"].values

    df_features["frequency_min"] = df_intermediate[["freq_a", "freq_b"]].min(axis=1).apply(lambda x: min(x,max_freq))
    df_features["frequency_max"] = df_intermediate[["freq_a", "freq_b"]].max(axis=1).apply(lambda x: min(x,max_freq))

    print("--> Compute neighbor features...")
    for i in range(1, max_level+1):
        df_intermediate["neighbors_a"] = df_questions.loc[df.text_a_ID, "neighbors" + str(i)].values
        df_intermediate["neighbors_b"] = df_questions.loc[df.text_b_ID, "neighbors" + str(i)].values
        df_intermediate["common_neighbors"] = df_intermediate.apply(lambda x: len(list(x.neighbors_a.intersection(x.neighbors_b))), axis=1)

        df_features["common_neighbors" + str(i)] = df_intermediate["common_neighbors"].apply(lambda x: min(x,max_neighbors)).apply(lambda x: min(x,max_neighbors))
        df_intermediate["min_neighbors"] = df_intermediate[["neighbors_a","neighbors_b"]].apply(lambda x: min(len(x.neighbors_a), len(x.neighbors_b)),axis=1)
        df_features["common_neighbors_ratio" + str(i)] = df_features["common_neighbors" + str(i)]/(df_intermediate["min_neighbors"] + 0.00001)

    print("--> Compute k-core features...")
    df_intermediate["k_core_a"] = df_questions.loc[df.text_a_ID, "k_core"].values
    df_intermediate["k_core_b"] = df_questions.loc[df.text_b_ID, "k_core"].values

    df_features["k_core_min".format(i)] = df_intermediate[["k_core_a", "k_core_b"]].min(axis=1)
    df_features["k_core_max".format(i)] = df_intermediate[["k_core_a", "k_core_b"]].max(axis=1)
    
    print("--> Compute page rank features...")
    df_intermediate["page_rank_a"] = df_questions.loc[df.text_a_ID, "page_rank"].values
    df_intermediate["page_rank_b"] = df_questions.loc[df.text_b_ID, "page_rank"].values

    df_features["page_rank_min"] = df_intermediate[["page_rank_a", "page_rank_b"]].min(axis=1).apply(lambda x: min(x,100))
    df_features["page_rank_max"] = df_intermediate[["page_rank_a", "page_rank_b"]].max(axis=1).apply(lambda x: min(x,100))
    
    print("--> Compute closeness centrality features...")
    df_intermediate["closeness_centrality_a"] = df_questions.loc[df.text_a_ID, "closeness_centrality"].values
    df_intermediate["closeness_centrality_b"] = df_questions.loc[df.text_b_ID, "closeness_centrality"].values

    df_features["closeness_centrality_min"] = df_intermediate[["closeness_centrality_a", "closeness_centrality_b"]].min(axis=1).apply(lambda x: min(x,100))
    df_features["closeness_centrality_max"] = df_intermediate[["closeness_centrality_a", "closeness_centrality_b"]].max(axis=1).apply(lambda x: min(x,100))
    
    print("--> Compute clustering features...")
    df_intermediate["clustering_a"] = df_questions.loc[df.text_a_ID, "clustering"].values
    df_intermediate["clustering_b"] = df_questions.loc[df.text_b_ID, "clustering"].values

    df_features["clustering_min"] = df_intermediate[["clustering_a", "clustering_b"]].min(axis=1).apply(lambda x: min(x,100))
    df_features["clustering_max"] = df_intermediate[["clustering_a", "clustering_b"]].max(axis=1).apply(lambda x: min(x,100))
    
    print("--> Compute eigenvector centrality...")
    df_intermediate["eigenvector_centrality_a"] = df_questions.loc[df.text_a_ID, "eigenvector_centrality"].values
    df_intermediate["eigenvector_centrality_b"] = df_questions.loc[df.text_b_ID, "eigenvector_centrality"].values

    df_features["eigenvector_centrality_min"] = df_intermediate[["eigenvector_centrality_a", "eigenvector_centrality_b"]].min(axis=1).apply(lambda x: min(x,100))
    df_features["eigenvector_centrality_max"] = df_intermediate[["eigenvector_centrality_a", "eigenvector_centrality_b"]].max(axis=1).apply(lambda x: min(x,100))
    
    
    return df_features

print("Compute train features...")
train_features = preprocess(train)

print("Compute test features...")
test_features = preprocess(test)


train_features.to_csv("non_nlp_features_train.csv", index=False)
test_features.to_csv("non_nlp_features_test.csv", index=False)