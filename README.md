# Sentiment Classification models
## This repository consists of two models:
  * An Embedding Layer + LSTM 
  * TF-IDF with unigrams, bigrams and trigrams + Logistic Regression

## Details of Files in the repository:
* LSTM Model
  * classifier.py - Contains Pre-processing, model definition, model training, performance plotting and checkpointing best model
  * predict.py - Contains code for live lesting the classifier
  * tokenizer.json - Contains the trained tokens on the IMDB dataset
  * best_model_1.h5 - Contains the model weights of the best performing model
  

* TF-IDF + LR Model
  * tfidf_ngram_classifier.py - Contains model definition, model training and model saving
  * tfidf_ngram_predict.py - Contains code for live lesting the classifier
  
  ## To test the models:  
  * LSTM Model
    Clone the repository into your local system and run the following command:
    
    ```Python predict.py``` 
    
   * TF-IDF + LR Model
   Clone the repository into your local system and run the following commands:
   
     ```python tfidf_ngram_classifier.py```
   
    ```python tfidf_ngram_predict.py```
   
  
  ## Running train the models:
  
  Place the IMDB dataset in the same location as the classifier files. 
   * LSTM Model
    Clone the repository into your local system and run the following command:
    
     ```python classifier.py```
    
     ```Python predict.py``` 
    
   * TF-IDF + LR Model
   Clone the repository into your local system and run the following commands:
    
     ```python tfidf_ngram_classifier.py```
   
      ```python tfidf_ngram_predict.py```
  
  
  
