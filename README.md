# Twitter NLP Sentiment Analysis
Machine learning project using a Twitter tweets dataset (Sentiment140).

## Description

The Twitter NLP Sentiment Analysis project aims to classify tweets into positive or negative sentiments using Natural Language Processing (NLP) techniques and machine learning models. By preprocessing the text data and training classifiers such as Logistic Regression, Bernoulli Naive Bayes, and Linear Support Vector Machines (SVM), the project determines sentiment polarity to assist in social media sentiment analysis.

The project processes the Sentiment140 dataset, consisting of tweets labeled for positive and negative sentiments, and applies feature extraction and classification methods to evaluate sentiment performance.

## About the project:

### Programming Languages / Frameworks

* Python
* Libraries:
  
* Scikit-learn
* NLTK (Natural Language Toolkit)
* Matplotlib
* Seaborn

### Features 

1. Preprocessing:
   * Cleaning text data by removing stopwords, punctuation, URL's, and numbers
   * Tokenization, stemming, and lemmatization to standardize textual data.

2. Feature Extraction:
   * Using TF-IDF Vectorizer to convert text into numerical representations with n-gram support.

3. Models:
   * Logistic Regression
   * Bernoulli Naive Bayes
   * Linear Support Vector Machines
     
4.  Evaluation:
   * Classification reports
   * Confusion matrices
   * Receiver Operating Characteristic (ROC) curve plots

### How to run:

1. Install Python 3.8 or higher.
2. Install the required libraries by running:

```
pip install numpy pandas matplotlib seaborn nltk scikit-learn wordcloud
```
Steps:

1. Clone or download the repository.
2. Place the dataset file (twitter_data140.csv) in the same directory.
3. Run the script in your Python environment:

```
python twitternlp.py
```

## Contributors
- Amelie Gomez (agrnerd17)
- Robert Moberly (rmoberly91)
