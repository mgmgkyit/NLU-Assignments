# Assignment A1 : That’s What I LIKE
Student ID: st125326

## How to run the web app
from the command prompt, run : "python app/app.py"
Access the app from http://127.0.0.1:5000

## How to use website
1. Enter one word to search for similarities in the search bar.
2. The program will run and display the top 10 most similar words from each model in tableformat. 
There are four approaches [ Word2Vec (Skipgram), Word2Vec Skipgram (Neg Sampling), GloVe from Scratch, and GloVe (Gensim) ]

## Training Data
1. Corpus source - nltk datasets('reuters'), categories is "coffee"
2. Data source: https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html
3. Download Link in NLTK: https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/reuters.zip 


## Model Comparison

| Model                | Window Size       | Training Loss   | Training Time     | Syntactic Accuracy | Semantic Accuracy |
|----------------------|-------------------|-----------------|-------------------|--------------------|-------------------|
| Skipgram             | 2                 | 8.395           |  1 Mins 34 Secs   | 0%                 | 0%                |
| Skipgram (NEG)       | 2                 | 1.215           |  2 Mins 29 Secs   | 0%                 | 0%                |
| Glove from Scratch   | 2                 | 3.055           |  1 Mins 37 Secs   | 0%                 | 0%                |
| Glove (Gensim)       | Default           | -               | -                 | 4.66%              | 48.09%            |


## Similarity Scores (Spearman correlation:)
| Model                | Skipgram  | NEG      | GloVe    | GloVe (Gensim) |
|----------------------|-----------|----------|----------|----------------|
| MSE                  | -0.1402   | 0.0537   | -0.2576  |   0.5439       |

