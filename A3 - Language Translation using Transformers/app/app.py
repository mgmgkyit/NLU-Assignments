import torch
import torchtext
import pickle
import os
import numpy as np

from flask import Flask, render_template, request
from models.classes import *
from templates.library import pyicu_tokenizer, SRC_LANGUAGE, TRG_LANGUAGE, get_text_transform

# Set the default directory to the current directory ***
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Importing training data back into the program
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Define device

# Importing training data back into the program
meta = pickle.load(open('models/meta_additive.pkl', 'rb'))

# Define Transformers
# Define special symbols and indices
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

token_transform = meta['token_transform']
vocab_transform = meta['vocab_transform']
text_transform = get_text_transform(token_transform, vocab_transform)


# Initialize Flask app
app = Flask(__name__)

# Model Parameters and loading

input_dim   = len(vocab_transform[SRC_LANGUAGE])
output_dim  = len(vocab_transform[TRG_LANGUAGE])

hid_dim = 256
enc_layers = 3
dec_layers = 3
enc_heads = 8
dec_heads = 8
enc_pf_dim = 512
dec_pf_dim = 512
enc_dropout = 0.1
dec_dropout = 0.1

SRC_PAD_IDX = PAD_IDX
TRG_PAD_IDX = PAD_IDX

enc = Encoder(input_dim, 
              hid_dim, 
              enc_layers, 
              enc_heads, 
              enc_pf_dim, 
              enc_dropout, 
              device)

dec = Decoder(output_dim, 
              hid_dim, 
              dec_layers, 
              dec_heads, 
              dec_pf_dim, 
              enc_dropout, 
              device)

model = Seq2SeqTransformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
model.apply(initialize_weights)

# End model parameters and loading


# Define a route and a function that handles requests to it
@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/translate', methods=['POST'])
def translate():

    print(' ', model.load_state_dict(torch.load("models/Seq2SeqTransformer_additive.pt", map_location=device)))

    max_seq = 500

    # get prompt from HTML form.
    prompt = request.form['query'].strip()
    
    src_text = text_transform[SRC_LANGUAGE](prompt).to(device)
    src_text = src_text.reshape(1, -1)
    text_length = torch.tensor([src_text.size(0)]).to(dtype=torch.int64)
    src_mask = model.make_src_mask(src_text)

    model.eval()
    with torch.no_grad():
        enc_output = model.encoder(src_text, src_mask)
    
    outputs = []
    input_tokens = [EOS_IDX]
    for i in range(max_seq):
        with torch.no_grad():
            starting_token = torch.LongTensor(input_tokens).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(starting_token)

            output, attention = model.decoder(starting_token, enc_output, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()
        input_tokens.append(pred_token)
        outputs.append(pred_token)
        
        if pred_token == EOS_IDX:
            break
    
    print(outputs)
    trg_tokens = [vocab_transform[TRG_LANGUAGE].get_itos()[i] for i in outputs]

    translated_text = " ".join(trg_tokens[1:-1])

    return render_template('index.html', result = translated_text, old_query = prompt)

                               
# Run the app if the script is executed directly
if __name__ == '__main__':
    app.run(debug=True)

# End of file