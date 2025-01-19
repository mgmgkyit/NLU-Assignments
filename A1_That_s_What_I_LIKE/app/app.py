import torch
import json
import torch.nn as nn
import pickle

from flask import Flask, render_template, request
from fuzzywuzzy import process
from numpy import dot
from numpy.linalg import norm
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from models.classes import Skipgram, SkipgramNeg, Glove
import os

# Set the default directory to the current directory ***
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask app
app = Flask(__name__)

# Downloadd the four pre-trained models from disk
# Loading Models 

# skipgram
skg_args = pickle.load(open('models/skipgram.args', 'rb'))
word2index = skg_args.get('word2index', {})
vocab_skipgram = list(skg_args.get('word2index', {}).keys())
skg_args.pop('word2index', None) 
model_skipgram = Skipgram(**skg_args)
model_skipgram.load_state_dict(torch.load('models/skipgram.model', weights_only=True))

#SkipgramNeg
neg_args = pickle.load(open('models/neg.args', 'rb'))
vocab_neg = list(neg_args.get('word2index', {}).keys())
neg_args.pop('word2index', None) 
model_neg = SkipgramNeg(**neg_args)
model_neg.load_state_dict(torch.load('models/neg.model', weights_only=True))

#Glove
glove_args = pickle.load(open('models/glove.args', 'rb'))
glove_args.pop('word2index', None) 
model_glove = Glove(**glove_args)
model_glove.load_state_dict(torch.load('models/glove.model', weights_only=True))

#Gensim
glove_file = datapath('glove.6B.100d.txt')
gensim_model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
vocab_glove = gensim_model.index_to_key

# define model names
models = {
    'Skipgram': model_skipgram,
    'SkipgramNeg': model_neg,
    'Glove': model_glove
}

# Combine vocabularies (remove duplicates)
vocab = list(set(vocab_skipgram + vocab_neg + vocab_glove))


# Cosine similarity function
def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# Function to correct misspelled words
def correct_spelling(word, vocab, threshold=80):

    # Find the best match in the vocabulary
    match, score = process.extractOne(word, vocab)
    
    # Return the corrected word if the score is above the threshold
    if score >= threshold:
        return match
    else:
        return word  # Return the original word if no good match is found

# Function to get the top 10 similar words
def get_top_similar_words(model, word_input):

    try:
        if len(word_input.split()) == 1:  # Ensure input is a single word
            # Correct the spelling of the input word
            corrected_word = correct_spelling(word_input, vocab)
            
            # If the corrected word is different, notify the user
            if corrected_word != word_input:
                print(f"Corrected '{word_input}' to '{corrected_word}'")
            
            # Get the word embedding of the corrected word
            word_embed = model.get_vector(corrected_word).detach().numpy().flatten()
            similarity_dict = {}

            # Compute cosine similarity for each word in the vocabulary
            for a in vocab:
                try:
                    a_embed = model.get_vector(a).detach().numpy().flatten()
                    similarity_dict[a] = cos_sim(word_embed, a_embed)
                except KeyError:
                    continue  # Skip words not in the model's vocabulary

            # Sort the dictionary by similarity in descending order
            similarity_dict_sorted = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)

            # Return top 10 similar words
            return [f"{i+1}. {similarity_dict_sorted[i][0]} ({similarity_dict_sorted[i][1]:.8f})" for i in range(10)]
        else:
            return ["The system can search this word."]
    except KeyError:
        return ["The word is not found. Please enter a new word."]


@app.route('/', methods=['GET', 'POST'])
def index():
    search_query = None
    glove_output = []
    gensim_output = []
    skipgram_output = []
    skipgram_neg_output = []

    if request.method == 'POST':
        search_query = request.form['search_query']
        
        model_outputs = {}

        for model_name, model in models.items():
            try:
                model_outputs[model_name] = get_top_similar_words(model, search_query)
            except KeyError as e:
                model_outputs[model_name] = [str(e)]  # Handle unknown words
        
        # Assign outputs to respective variables
        glove_output = model_outputs.get("Glove", ["No results available."])
        skipgram_output = model_outputs.get("Skipgram", ["No results available."])
        skipgram_neg_output = model_outputs.get("SkipgramNeg", ["No results available."])
        gensim_output = gensim_model.most_similar(search_query, topn=10)
        gensim_output = [f"{i+1}. {word} ({similarity:.4f})" for i, (word, similarity) in enumerate(gensim_output)]

    return render_template('index.html', search_query=search_query, 
                           glove_output=glove_output, gensim_output=gensim_output, 
                           skipgram_output=skipgram_output, skipgram_neg_output=skipgram_neg_output)

if __name__ == '__main__':
    app.run(debug=True)
