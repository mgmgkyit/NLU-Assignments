from flask import Flask, render_template, request
import torch
from utils import *
import os

app = Flask(__name__)

# Set the default directory to the current directory ***
os.chdir(os.path.dirname(os.path.abspath(__file__)))

@app.route('/')
def home():
    return render_template('index.html')

with open('model/bert.param', 'rb') as f:
    bert_param = pickle.load(f)
    word2id = bert_param['word2id']

vocab_size = len(word2id)
max_len = 1000
n_layers = 6
n_heads = 8
d_model = 768
d_ff = d_model * 4
d_k = d_v = 64
n_segments = 2

model = BERT(
    n_layers=n_layers,
    n_heads=n_heads,
    d_model=d_model,
    d_ff=d_ff,
    d_k=d_k,
    n_segments=n_segments,
    vocab_size=vocab_size,
    max_len=max_len,
    device=device
).to(device)
model.load_state_dict(torch.load('model/bert_model.pt'))

@app.route('/submit', methods=['POST'])
def submit():
    premise = request.form['premise']
    hypothesis = request.form['hypothesis']
    
    # Simple example of result generation logic
    similarity = calculate_similarity(model, tokenizer, premise, hypothesis, device)
    similarity = round(similarity, 4)
    
    # Display classification result based on similarity score
    if similarity > 0.7:
        label = "Entailment"
    elif similarity > 0.3:
        label = "Neutral"
    else:
        label = "Contradiction"
        
    return render_template('index.html', premise=premise, hypothesis=hypothesis, result=label)

if __name__ == '__main__':
    app.run(debug=True)
