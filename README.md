# Prediction of semantically equivalent queries

This is the repository for the final project of the course INF582 - Introduction to Text Mining and NLP (2018-2019)

### Authors:
Fabrizio Indirli, Dor Polikar, Simon Klotz

## Instructions:
The code can be run using the following steps: <br>

### Getting the data:
1. Copy the train.csv and test.csv files into the data folder  
2. Generate or copy GloVe vectors:  
	a. If not already done, download the GloVe *840B-300d* file from [here](http://nlp.stanford.edu/data/glove.840B.300d.zip), put it in __/data/__ and convert it to word2vec format:  
	```
	python -m gensim.scripts.glove2word2vec --input  ./data/glove.840B.300d.txt --output ./data/glove.840B.300d.w2vformat.txt
	```
	b. Otherwise copy already converted glove file *glove.840B.300d.txt* to __./data/__ and rename it to   *glove.840B.300d.w2vformat.txt*

### Install required packages:
``` pip install -r requirements.txt ```

### Computing the features:
Run: ``` python ./build_features.py ``` 

### Predicting:

#### To get the results using the LSTM:
Run: ```python ./lstm_model.py```
 
The final submission is in the __predictions__ folder and called *postprocessed_submission.csv*

#### To get the results using the ensemble (if the ensemble should include the LSTM first run the lstm_model.py):  
Run: ```python ./cross_validation_ensemble.py```  
The final submission is in the __predictions__ folder and called *postprocessed_submission.csv*
