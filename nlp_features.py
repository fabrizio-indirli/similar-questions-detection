import pandas as pd
import numpy as np

import re
import string

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from fuzzywuzzy import fuzz
import distance

train = pd.read_csv("./train.csv", names=['row_ID', 'text_a_ID', 'text_b_ID', 'text_a_text', 'text_b_text', 'have_same_meaning'], index_col=0)
test = pd.read_csv("./test.csv", names=['row_ID', 'text_a_ID', 'text_b_ID', 'text_a_text', 'text_b_text', 'have_same_meaning'], index_col=0)
submission_sample = pd.read_csv("./sample_submission_file.csv")
en_stop = set(stopwords.words('english'))

def clean(q):
    # Adapted from https://github.com/aerdem4/kaggle-quora-dup
    q = str(q).lower()
    q = q.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")                           .replace("€", " euro ").replace("'ll", " will")
    q = re.sub(r"([0-9]+)000000", r"\1m", q)
    q = re.sub(r"([0-9]+)000", r"\1k", q)
    
    return q

def get_longest_substring_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    return 0 if len(strs) == 0 else len(strs[0]) / (min(len(a), len(b)) + 1)

def has_number(s):
    return int(any(char.isdigit() for char in s)             
               or any (x in s for x in ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                                    "eleven", "twelve", "thirteen", "fourteen"]))

def n_capital_letters(series):
    return series.apply(lambda x: sum(1 for c in x if c.isupper()))

def preprocess(df):
    df_features = pd.DataFrame(index=df.index)
    df_intermediate = pd.DataFrame(index=df.index)

    print("--> Compute tokens...")
    df_intermediate["clean_a"] = df.text_a_text.apply(lambda x: clean(x))
    df_intermediate["clean_b"] = df.text_b_text.apply(lambda x: clean(x))
    
    df_intermediate["words_a"] = df_intermediate.apply(lambda row: row.clean_a.split(" "), axis=1)
    df_intermediate["words_b"] = df_intermediate.apply(lambda row: row.clean_b.split(" "), axis=1)

    df_intermediate["words_clean_a"] = df_intermediate.apply(lambda row: set([w for w in row.words_a if w not in en_stop]), axis=1)
    df_intermediate["words_clean_b"] = df_intermediate.apply(lambda row: set([w for w in row.words_b if w not in en_stop]), axis=1)

    df_intermediate["stop_a"] = df_intermediate.apply(lambda row: set([w for w in row.words_a if w in en_stop]), axis=1)
    df_intermediate["stop_b"] = df_intermediate.apply(lambda row: set([w for w in row.words_b if w in en_stop]), axis=1)
    
    print("--> Compute common words features...")
    df_intermediate["common_stop_words"] = df_intermediate.apply(lambda row: row.stop_a.intersection(row.stop_b), axis=1)
    df_intermediate["common_words"] = df_intermediate.apply(lambda row: set(row.words_a).intersection(set(row.words_b)), axis=1)
    df_intermediate["common_clean_words"] = df_intermediate.apply(lambda row: row.words_clean_a.intersection(row.words_clean_b), axis=1)

    df_intermediate["common_stop_words_cnt"] = df_intermediate.common_stop_words.apply(lambda x: len(x))
    df_intermediate["common_words_cnt"] = df_intermediate.common_words.apply(lambda x: len(x))
    df_intermediate["common_clean_words_cnt"] = df_intermediate.common_clean_words.apply(lambda x: len(x))
    
    df_features["common_stop_words_ratio_min"] = df_intermediate.apply(lambda x: x.common_stop_words_cnt / (min(len(x["stop_a"]), len(x["stop_b"]))+0.0001), axis=1)
    df_features["common_words_ratio_min"] = df_intermediate.apply(lambda x: x.common_words_cnt / (min(len(x["words_a"]), len(x["words_b"]))+0.0001), axis=1)
    df_features["common_clean_words_ratio_min"] = df_intermediate.apply(lambda x: x.common_clean_words_cnt / (min(len(x["words_clean_a"]), len(x["words_clean_b"]))+0.0001), axis=1)

    df_features["common_stop_words_ratio_max"] = df_intermediate.apply(lambda x: x.common_stop_words_cnt / (max(len(x["stop_a"]), len(x["stop_b"]))+0.0001), axis=1)
    df_features["common_words_ratio_max"] = df_intermediate.apply(lambda x: x.common_words_cnt / (max(len(x["words_a"]), len(x["words_b"]))+0.0001), axis=1)
    df_features["common_clean_words_ratio_max"] = df_intermediate.apply(lambda x: x.common_clean_words_cnt / (max(len(x["words_clean_a"]), len(x["words_clean_b"]))+0.0001), axis=1)
    
    print("--> Compute general NLP features...")    
    df_features["same_last_token"] = df_intermediate.apply(lambda x: int(x.words_a[-1] == x.words_b[-1]), axis=1)
    df_features["same_first_token"] = df_intermediate.apply(lambda x: int(x.words_a[0] == x.words_b[0]), axis=1)
    
    df_features["length_diff"] = df_intermediate.apply(lambda x: abs(len(x.words_a) - len(x.words_b)), axis=1)
    df_features["avg_length"] = df_intermediate.apply(lambda x: (len(x.words_a) + len(x.words_b))/2, axis=1)
    
#     # Number of capital letters feature
#     df_intermediate["a_n_capital"] = n_capital_letters(df["text_a_text"])
#     df_intermediate["b_n_capital"] = n_capital_letters(df["text_b_text"])
#     df_features["max_n_capital"] = df_intermediate[["a_n_capital", "b_n_capital"]].max(axis=1)
#     df_features["min_n_capital"] = df_intermediate[["a_n_capital", "b_n_capital"]].min(axis=1)
#     df_features["n_capital_diff"] = np.abs(df_intermediate["a_n_capital"] - df_intermediate["b_n_capital"])
    
#     # Number related features
#     df_intermediate["a_has_number"] = df.text_a_text.apply(lambda x: has_number(x))
#     df_intermediate["b_has_number"] = df.text_b_text.apply(lambda x: has_number(x))
#     df_features["max_has_number"] = df_intermediate[["a_has_number", "b_has_number"]].max(axis=1)
#     df_features["min_has_number"] = df_intermediate[["a_has_number", "b_has_number"]].min(axis=1)
    
    print("--> Compute fuzzy features...") 
    #df_features['fuzz_qratio'] = df.apply(lambda x: fuzz.QRatio(str(x["text_a_text"]), str(x["text_b_text"])), axis=1)
    df_features['fuzz_WRatio'] = df.apply(lambda x: fuzz.WRatio(str(x["text_a_text"]), str(x["text_b_text"])), axis=1)
    df_features['fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(str(x["text_a_text"]), str(x["text_b_text"])), axis=1)
    #df_features['fuzz_partial_token_set_ratio'] = df.apply(lambda x: fuzz.partial_token_set_ratio(str(x["text_a_text"]), str(x["text_b_text"])), axis=1)
    #df_features['fuzz_partial_token_sort_ratio'] = df.apply(lambda x: fuzz.partial_token_sort_ratio(str(x["text_a_text"]), str(x["text_b_text"])), axis=1)
    df_features['fuzz_token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(str(x["text_a_text"]), str(x["text_b_text"])), axis=1)
    df_features['fuzz_token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(str(x["text_a_text"]), str(x["text_b_text"])), axis=1)

    print("--> Compute longest substring...") 
    df_features["longest_substring_ratio"]  = df.apply(lambda x: get_longest_substring_ratio(x["text_a_text"], x["text_b_text"]), axis=1)
    
    return df_features


print("Compute train features...")
train_features = preprocess(train)

print("Compute test features...")
test_features = preprocess(test)

train_features.to_csv("nlp_features_train.csv", index=False)
test_features.to_csv("nlp_features_test.csv", index=False)