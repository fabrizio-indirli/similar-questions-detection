import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec, Word2Vec, TaggedDocument
from gensim.models import LsiModel
from gensim import corpora

from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances
import nltk
from nltk.corpus import stopwords

# Load data
train = pd.read_csv("./data/train.csv", names=['row_ID', 'text_a_ID', 'text_b_ID', 'text_a_text', 'text_b_text', 'have_same_meaning'], index_col=0)
test = pd.read_csv("./data/test.csv", names=['row_ID', 'text_a_ID', 'text_b_ID', 'text_a_text', 'text_b_text', 'have_same_meaning'], index_col=0)
submission_sample = pd.read_csv("./sample_submission_file.csv")
en_stop = set(stopwords.words('english'))
glove_file = "./data/word2vec.glove.6B.100d.txt"


def clean(q):
    # Adapted from https://github.com/aerdem4/kaggle-quora-dup
    q = str(q).lower()
    q = q.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")                           .replace("€", " euro ").replace("'ll", " will")
    q = re.sub(r"([0-9]+)000000", r"\1m", q)
    q = re.sub(r"([0-9]+)000", r"\1k", q)
    return q

# Start computation
all_questions = pd.concat([train["text_a_text"], train["text_b_text"], test["text_a_text"], test["text_b_text"]])
question_counts = all_questions.value_counts()

questions = [clean(q) for q in all_questions]
questions_token = [[w for w in q.split(' ') if w not in en_stop] for q in questions]


print("Fit TFIDF Model...")
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))
tfidf_vectorizer.fit(all_questions);

print("Load Glove Model...")
glove_model = KeyedVectors.load_word2vec_format(glove_file)

print("Fit Word2Vec Model...")
word2Vec = Word2Vec(size=100, window=5, min_count=2, sg=1, workers=10)
word2Vec.build_vocab(questions_token)  # prepare the model vocabulary
word2Vec.train(sentences=questions_token, total_examples=len(questions_token), epochs=word2Vec.iter)

print("Fit LSI Model...")
dictionary = corpora.Dictionary(questions_token)
corpus = [dictionary.doc2bow(text) for text in questions_token]
lsi = LsiModel(corpus, num_topics=200, id2word = dictionary)

print("Fit doc2vec Model...")
q2idx_dict = {tuple(q): idx for idx, q in enumerate(questions_token)}

d2v_training_data = []
for idx,doc in enumerate(questions_token):
    d2v_training_data.append(TaggedDocument(doc,[idx]))

d2v_dm = Doc2Vec(d2v_training_data, size=100, window=4, min_count=3, workers=16, iter=5)
d2v_dm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

d2v_bow = Doc2Vec(d2v_training_data, size=100, window=4, min_count=3, dm=0, workers=16, iter=5)
d2v_bow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

def preprocess(df):
    df_features = pd.DataFrame(index=df.index)
    df_intermediate = pd.DataFrame(index=df.index)

    print("--> Compute tokens...")
    df_intermediate["clean_a"] = df.text_a_text.apply(lambda x: clean(x))
    df_intermediate["clean_b"] = df.text_b_text.apply(lambda x: clean(x))

    df_intermediate["words_a"] = df_intermediate.apply(lambda row: row.clean_a.split(" "), axis=1)
    df_intermediate["words_b"] = df_intermediate.apply(lambda row: row.clean_b.split(" "), axis=1)

    df_intermediate["words_clean_a"] = df_intermediate.apply(lambda row: [w for w in row.words_a if w not in en_stop], axis=1)
    df_intermediate["words_clean_b"] = df_intermediate.apply(lambda row:  [w for w in row.words_b if w not in en_stop], axis=1)

    df_intermediate["stop_a"] = df_intermediate.apply(lambda row: [w for w in row.words_a if w in en_stop], axis=1)
    df_intermediate["stop_b"] = df_intermediate.apply(lambda row: [w for w in row.words_b if w in en_stop], axis=1)
    
    print("--> Compute tfidf distance...")
    tfidf_a = tfidf_vectorizer.transform(df_intermediate["clean_a"])
    tfidf_b = tfidf_vectorizer.transform(df_intermediate["clean_b"])

    df_features["tfidf_dist_cosine"] = paired_cosine_distances(tfidf_a, tfidf_b)
    df_features["tfidf_dist_euclidean"] = paired_euclidean_distances(tfidf_a, tfidf_b)
    
    print("--> Compute glove distance...")
    glove_emb_a = df_intermediate["words_clean_a"].apply(lambda q: np.array([glove_model.wv[w] for w in q if w in glove_model.wv]))
    glove_emb_b = df_intermediate["words_clean_b"].apply(lambda q: np.array([glove_model.wv[w] for w in q if w in glove_model.wv]))

    glove_emb_a[glove_emb_a.apply(lambda x: len(x)==0)] = glove_emb_a[glove_emb_a.apply(lambda x: len(x)==0)].apply(lambda y: np.zeros((1,100)))
    glove_emb_b[glove_emb_b.apply(lambda x: len(x)==0)] = glove_emb_b[glove_emb_b.apply(lambda x: len(x)==0)].apply(lambda y: np.zeros((1,100)))
    glove_emb_a = glove_emb_a.apply(lambda x: np.mean(x,axis=0))
    glove_emb_b = glove_emb_b.apply(lambda x: np.mean(x,axis=0))
    glove_emb_a = np.vstack(glove_emb_a.values)
    glove_emb_b = np.vstack(glove_emb_b.values)

    df_features["glove_dist_cosine"] = paired_cosine_distances(glove_emb_a, glove_emb_b)
    df_features["glove_dist_euclidean"] = paired_euclidean_distances(glove_emb_a, glove_emb_b)
    df_features["glove_word_mover_dist"] = df_intermediate.apply(lambda row: glove_model.wv.wmdistance(row["words_clean_a"], row["words_clean_b"]), axis=1)
    
    print("--> Compute lsi distance...")
    lsi_emb_a = df_intermediate["words_clean_a"].apply(lambda x: np.array(lsi[dictionary.doc2bow(x)]))
    lsi_emb_b = df_intermediate["words_clean_b"].apply(lambda x: np.array(lsi[dictionary.doc2bow(x)]))

    lsi_emb_a[lsi_emb_a.apply(lambda x: len(x)==0 or x.shape[0]!=200)] = lsi_emb_a[lsi_emb_a.apply(lambda x: len(x)==0 or x.shape[0]!=200)].apply(lambda x: np.zeros((200,2)))
    lsi_emb_b[lsi_emb_b.apply(lambda x: len(x)==0 or x.shape[0]!=200)] = lsi_emb_b[lsi_emb_b.apply(lambda x: len(x)==0 or x.shape[0]!=200)].apply(lambda x: np.zeros((200,2)))

    # Derive question representations from single lsi vectors
    lsi_emb_a = lsi_emb_a.apply(lambda x: np.mean(x,axis=0))
    lsi_emb_b = lsi_emb_b.apply(lambda x: np.mean(x,axis=0))
    lsi_emb_a = np.vstack(lsi_emb_a.values)
    lsi_emb_b = np.vstack(lsi_emb_b.values)

    df_features["lsi_dist_cosine"] = paired_cosine_distances(lsi_emb_a, lsi_emb_b)
    df_features["lsi_dist_euclidean"] = paired_euclidean_distances(lsi_emb_a, lsi_emb_b)
    
    print("--> Compute word2vec distance...")
    word2Vec_emb_a = df_intermediate["words_clean_a"].apply(lambda q: np.array([word2Vec.wv[w] for w in q if w in word2Vec.wv]))
    word2Vec_emb_b = df_intermediate["words_clean_b"].apply(lambda q: np.array([word2Vec.wv[w] for w in q if w in word2Vec.wv]))

    word2Vec_emb_a[word2Vec_emb_a.apply(lambda x: len(x)==0)] = word2Vec_emb_a[word2Vec_emb_a.apply(lambda x: len(x)==0)].apply(lambda y: np.zeros((1,100)))
    word2Vec_emb_b[word2Vec_emb_b.apply(lambda x: len(x)==0)] = word2Vec_emb_b[word2Vec_emb_b.apply(lambda x: len(x)==0)].apply(lambda y: np.zeros((1,100)))

    word2Vec_emb_a = word2Vec_emb_a.apply(lambda x: np.mean(x,axis=0))
    word2Vec_emb_b = word2Vec_emb_b.apply(lambda x: np.mean(x,axis=0))
    word2Vec_emb_a = np.vstack(word2Vec_emb_a.values)
    word2Vec_emb_b = np.vstack(word2Vec_emb_b.values)

    df_features["w2v_dist_cosine"] = paired_cosine_distances(word2Vec_emb_a, word2Vec_emb_b)
    df_features["w2v_dist_euclidean"] = paired_euclidean_distances(word2Vec_emb_a, word2Vec_emb_b)
    df_features["word2vec_word_mover_dist"] = df_intermediate.apply(lambda row: word2Vec.wv.wmdistance(row["words_clean_a"], row["words_clean_b"]), axis=1)
    
    print("--> Compute doc2vec distance...")
    doc_vec_dm_emb_a = df_intermediate["words_clean_a"].apply(lambda q: d2v_dm.docvecs[q2idx_dict[tuple(q)]])
    doc_vec_dm_emb_b = df_intermediate["words_clean_b"].apply(lambda q: d2v_dm.docvecs[q2idx_dict[tuple(q)]])
    doc_vec_bow_emb_a = df_intermediate["words_clean_a"].apply(lambda q: d2v_bow.docvecs[q2idx_dict[tuple(q)]])
    doc_vec_bow_emb_b = df_intermediate["words_clean_b"].apply(lambda q: d2v_bow.docvecs[q2idx_dict[tuple(q)]])
    doc_vec_dm_emb_a = np.vstack(doc_vec_dm_emb_a.values)
    doc_vec_dm_emb_b = np.vstack(doc_vec_dm_emb_b.values)
    doc_vec_bow_emb_a = np.vstack(doc_vec_bow_emb_a.values)
    doc_vec_bow_emb_b = np.vstack(doc_vec_bow_emb_b.values)
        
    df_features["dm_dist_cosine"] = paired_cosine_distances(doc_vec_dm_emb_a, doc_vec_dm_emb_b)
    df_features["dm_dist_euclidean"] = paired_euclidean_distances(doc_vec_dm_emb_a, doc_vec_dm_emb_b)
    df_features["dm_word_mover_dist"] = df_intermediate.apply(lambda row: d2v_dm.wv.wmdistance(row["words_clean_a"], row["words_clean_b"]), axis=1)
    
    df_features["bow_dist_cosine"] = paired_cosine_distances(doc_vec_bow_emb_a, doc_vec_bow_emb_b)
    df_features["bow_dist_euclidean"] = paired_euclidean_distances(doc_vec_bow_emb_a, doc_vec_bow_emb_b)
    df_features["bow_word_mover_dist"] = df_intermediate.apply(lambda row: d2v_bow.wv.wmdistance(row["words_clean_a"], row["words_clean_b"]), axis=1)
    
    print("--> Compute edit distance...")
    df_features["edit_distance"] = df_intermediate.apply(lambda x: nltk.edit_distance(x["clean_a"], x["clean_b"]), axis=1)
        
    return df_features


print("Compute train features...")
train_features = preprocess(train)

print("Compute test features...")
test_features = preprocess(test)

print("Store features...")
train_features.to_csv("./data/distance_features_train.csv", index=False)
test_features.to_csv("./data/distance_features_test.csv", index=False)