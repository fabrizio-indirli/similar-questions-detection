This is the repository for the final project of the course INF582 - Introduction to Text Mining and NLP (2018-2019)

Authors:
Brizio Indirli
Dor Polikar
Simon Klotz

Instructions:
The code can be run using the following steps:

Getting the data:
1. Copy the train.csv and test.csv files into the data folder
2a. If not already done, convert glove vectors from ./data/glove.840B.300d.txt to word2vec format:
	python -m gensim.scripts.glove2word2vec --input  ./data/glove.840B.300d.txt --output glove.840B.300d.w2vformat.txt
2b. Otherwise copy already converted glove file glove.840B.300d.txt to ./data/ and rename it to glove.840B.300d.w2vformat.txt

Computing the features:
3. Run: python ./nlp_features.py
4. Run: python ./distance_features.py
5. Run: python ./graph_features.py

To get the results using the LSTM:
6. Run: python ./lstm_model.py
7. Run: python ./postprocess_submission.py
The final submission is in the predictions folder and called postprocessed_submission.csv 

To get the results using the ensemble (if the ensemble should include the LSTM first run the lstm_model.py):
8. Run: python ./cross_validation_ensemble.py
9. Run: python ./postprocess_submission.py
The final submission is in the predictions folder and called postprocessed_submission.csv
