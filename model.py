#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import re

from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold

from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.layers.core import Lambda
from keras.layers.merge import concatenate, add, multiply
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint

from nltk.corpus import stopwords

np.random.seed(0)

min_occurence = 100
unknown = "memento"
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 30
BATCH_SIZE = 1025
n_folds=10


print("Load glove embedding...")
glove_file = "./data/word2vec.glove.840B.300d.txt"
glove_model = KeyedVectors.load_word2vec_format(glove_file)

print("Load data...")
train = pd.read_csv("./data/train.csv", names=['row_ID', 'text_a_ID', 'text_b_ID', 'text_a_text', 'text_b_text', 'have_same_meaning'], index_col=0)
test = pd.read_csv("./data/test.csv", names=['row_ID', 'text_a_ID', 'text_b_ID', 'text_a_text', 'text_b_text', 'have_same_meaning'], index_col=0)
submission_sample = pd.read_csv("./sample_submission_file.csv")
en_stop = set(stopwords.words('english'))


print("Load train features...")
train_nlp_features = pd.read_csv("data/nlp_features_train.csv")
train_non_nlp_features = pd.read_csv("data/non_nlp_features_train.csv")

print("Load test features...")
test_nlp_features = pd.read_csv("data/nlp_features_train.csv")
test_non_nlp_features = pd.read_csv("data/non_nlp_features_train.csv")


lemmatizer = WordNetLemmatizer()

def lemmatize(word, lemmatizer):
    if len(word) < 4:
        return word
    return lemmatizer.lemmatize(lemmatizer.lemmatize(word, "n"), "v")

def clean(q):
    # Adopted from https://github.com/aerdem4/kaggle-quora-dup
    q = q.lower().replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")         .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")         .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")         .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")         .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")         .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")         .replace("€", " euro ").replace("'ll", " will").replace("=", " equal ").replace("+", " plus ")
    q = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', q)
    q = re.sub(r"([0-9]+)000000", r"\1m", q)
    q = re.sub(r"([0-9]+)000", r"\1k", q)
    q = ' '.join([lemmatize(w, lemmatizer) for w in q.split()])
    return q

def get_model(embedding_matrix, nb_words, n_features):    
    embedding_layer = Embedding(nb_words,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)
    lstm_layer = LSTM(75, recurrent_dropout=0.2)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    features_input = Input(shape=(n_features,), dtype="float32")
    features_dense = BatchNormalization()(features_input)
    features_dense = Dense(200, activation="relu")(features_dense)
    features_dense = Dropout(0.2)(features_dense)

    addition = add([x1, y1])
    minus_y1 = Lambda(lambda x: -x)(y1)
    merged = add([x1, minus_y1])
    merged = multiply([merged, merged])
    merged = concatenate([merged, addition])
    merged = Dropout(0.4)(merged)

    merged = concatenate([merged, features_dense])
    merged = BatchNormalization()(merged)
    merged = GaussianNoise(0.1)(merged)

    merged = Dense(150, activation="relu")(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    out = Dense(1, activation="sigmoid")(merged)

    model = Model(inputs=[sequence_1_input, sequence_2_input, features_input], outputs=out)
    return model

def is_numeric(s):
    return any(i.isdigit() for i in s)

def preprocess(q):
    processed_q = []
    surplus_q = []
    numbers_q = []
    new_unknown = True
    for word in q.split()[::-1]:
        if word in top_words:
            processed_q = [word] + processed_q
            new_unknown = True
        elif word not in en_stop:
            if new_unknown:
                processed_q = [unknown] + processed_q
                new_unknown = False
            if is_numeric(word):
                numbers_q = [word] + numbers_q
            else:
                surplus_q = [word] + surplus_q
        else:
            new_memento = True
        if len(processed_q) == MAX_SEQUENCE_LENGTH:
            break
    return " ".join(processed_q), set(surplus_q), set(numbers_q)


print("Compute train set features and embedding...")
train["text_a_text_clean"] = train["text_a_text"].fillna("").apply(clean)
train["text_b_text_clean"] = train["text_b_text"].fillna("").apply(clean)

unique_questions = pd.Series(train["text_a_text_clean"] + train["text_b_text_clean"]).unique()
count_vectorizer = CountVectorizer(lowercase=True, token_pattern="\S+", min_df=min_occurence)
count_vectorizer.fit(unique_questions)

top_words = set(count_vectorizer.vocabulary_.keys())
top_words.add(unknown)

train_q_a_features = train["text_a_text_clean"].apply(preprocess)
train_q_b_features = train["text_b_text_clean"].apply(preprocess)

train_q_a = train_q_a_features.fillna("").apply(lambda x: x[0])
train_q_b = train_q_b_features.fillna("").apply(lambda x: x[0])

train_intermediate_df = pd.DataFrame(index=train.index)
train_intermediate_df["surplus_a"] = train_q_a_features.apply(lambda x: x[1])
train_intermediate_df["surplus_b"] = train_q_b_features.apply(lambda x: x[1])
train_intermediate_df["number_a"] = train_q_a_features.apply(lambda x: x[2])
train_intermediate_df["number_b"] = train_q_b_features.apply(lambda x: x[2])

train_features_df = pd.DataFrame(index=train.index)
train_features_df["surplus_intersection"] = train_intermediate_df.apply(lambda x: len(x.surplus_a.intersection(x.surplus_b)), axis=1)
train_features_df["surplus_union"] = train_intermediate_df.apply(lambda x: len(x.surplus_a.union(x.surplus_b)), axis=1)
train_features_df["number_intersection"] = train_intermediate_df.apply(lambda x: len(x.number_a.intersection(x.number_b)), axis=1)
train_features_df["number_union"] = train_intermediate_df.apply(lambda x: len(x.number_a.union(x.number_b)), axis=1)

features_train = np.hstack((train_nlp_features, train_non_nlp_features, train_features_df))
n_features = features_train.shape[1]

tokenizer = Tokenizer(filters="")
tokenizer.fit_on_texts(np.append(train_q_a, train_q_b))
word_index = tokenizer.word_index

train_q_a_padded = pad_sequences(tokenizer.texts_to_sequences(train_q_a), maxlen=MAX_SEQUENCE_LENGTH)
train_q_b_padded = pad_sequences(tokenizer.texts_to_sequences(train_q_a), maxlen=MAX_SEQUENCE_LENGTH)


print("Compute test set features and embedding...")
test["text_a_text_clean"] = test["text_a_text"].fillna("").apply(clean)
test["text_b_text_clean"] = test["text_b_text"].fillna("").apply(clean)

q_a_features_test = test["text_a_text_clean"].apply(preprocess)
q_b_features_test = test["text_b_text_clean"].apply(preprocess)

test_q_a = q_a_features_test.apply(lambda x: x[0])
test_q_b = q_b_features_test.apply(lambda x: x[0])

test_intermediate_df = pd.DataFrame(index=test.index)
test_intermediate_df["surplus_a"] = q_a_features_test.apply(lambda x: x[1])
test_intermediate_df["surplus_b"] = q_b_features_test.apply(lambda x: x[1])
test_intermediate_df["number_a"] = q_a_features_test.apply(lambda x: x[2])
test_intermediate_df["number_b"] = q_b_features_test.apply(lambda x: x[2])

test_features_df = pd.DataFrame(index=test.index)
test_features_df["surplus_intersection"] = test_intermediate_df.apply(lambda x: len(x.surplus_a.intersection(x.surplus_b)), axis=1)
test_features_df["surplus_union"] = test_intermediate_df.apply(lambda x: len(x.surplus_a.union(x.surplus_b)), axis=1)
test_features_df["number_intersection"] = test_intermediate_df.apply(lambda x: len(x.number_a.intersection(x.number_b)), axis=1)
test_features_df["number_union"] = test_intermediate_df.apply(lambda x: len(x.number_a.union(x.number_b)), axis=1)

features_test = np.hstack((test_nlp_features, test_non_nlp_features, train_features_df))

test_q_a_padded = pad_sequences(tokenizer.texts_to_sequences(test_q_a), maxlen=MAX_SEQUENCE_LENGTH)
test_q_b_padded = pad_sequences(tokenizer.texts_to_sequences(test_q_a), maxlen=MAX_SEQUENCE_LENGTH)


labels = np.array(train["have_same_meaning"])
nb_words = len(word_index) + 1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

for word, i in word_index.items():
    embedding_vector=None
    if word in glove_model.wv:
        embedding_vector = glove_model.wv[word]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

        
kfold = StratifiedKFold(n_splits=n_folds, shuffle=True)
model_count = 0

train_ensemble = pd.DataFrame(index=train.index)
test_ensemble = pd.DataFrame(index=test.index)

train_ensemble["pred_lstm"] = 0
test_ensemble["pred_lstm"] = 0


test_predictions = []
for train_indices, validation_indices in kfold.split(train["have_same_meaning"], train["have_same_meaning"]):
    train_fold_a = train_q_a_padded[train_indices]
    train_fold_b = train_q_b_padded[train_indices]
    train_fold_features = features_train[train_indices]
    train_fold_labels = labels[train_indices]

    val_fold_a = train_q_a_padded[validation_indices]
    val_fold_b = train_q_b_padded[validation_indices]
    val_fold_features = features_train[validation_indices]
    val_fold_labels = labels[validation_indices]
    
    model = get_model(embedding_matrix, nb_words, n_features)
    model.compile(loss="binary_crossentropy", optimizer="nadam")
    
    early_stopping = EarlyStopping(monitor="val_loss", patience=5)
    best_model_path = "best_model" + str(model_count) + ".h5"
    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)

    hist = model.fit([train_fold_a, train_fold_b, train_fold_features], train_fold_labels,
                     validation_data=([val_fold_a, val_fold_b, val_fold_features], val_fold_labels),
                     epochs=50, batch_size=BATCH_SIZE, shuffle=True,
                     callbacks=[early_stopping, model_checkpoint], verbose=1)

    model.load_weights(best_model_path)
    print(model_count, "validation loss:", min(hist.history["val_loss"]))
    
    train_pred = model.predict([val_fold_a, val_fold_b, val_fold_features], batch_size=BATCH_SIZE, verbose=1)
    test_pred = model.predict([test_q_a_padded, test_q_b_padded, features_test], batch_size=BATCH_SIZE, verbose=1)
    
    test_ensemble["pred_lstm"] = test_pred
    train_ensemble.loc[validation_indices,"pred_lstm"] = train_pred.ravel()

    submission = pd.DataFrame({"Id": test.index, "Score": test_pred.ravel()})
    submission.to_csv("predictions/preds" + str(model_count) + ".csv", index=False)
    test_predictions.append(test_pred.ravel())
    
    model_count += 1


test_ensemble.to_csv("./predictions/test_ensemble_lstm.csv", index=False)
train_ensemble.to_csv("./predictions/train_ensemble_lstm.csv", index=False)

print("Compute average ensemble...")
test_predictions = np.mean(test_predictions, axis=0)

submission = pd.DataFrame({"Id":test.index, "Score":test_predictions})
submission.to_csv("predictions/submission.csv", index=False)