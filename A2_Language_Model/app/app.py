import torch
import torch.nn as nn
import pickle
import os

from flask import Flask, render_template, request
from templates.classes import LSTMLanguageModel, generate

# Set the default directory to the current directory ***
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Importing training data back into the program
Data = pickle.load(open('models/Data.pkl', 'rb'))
output = 'models/best-val-lstm_lm_ian_fleming.pt'  # Output file name for pre-trained model

# prearing to load pre-trained model's state dictionary from preshared google drive
# if you would like to run the app without the state dictionary file, please run this code block
# or you can manually download the file from the link provided in the README.md
import gdown
file_id = '1vSoQKY5adKdb9FVN9TMDTDso8_qABI1H'  # Since file is 158MB, we will use gdown to download it
url = f'https://drive.google.com/uc?id={file_id}'
gdown.download(url, output, quiet=False)
## End of code block

vocab_size = Data['vocab_size']
emb_dim = Data['emb_dim']
hid_dim = Data['hid_dim']
num_layers = Data['num_layers']
dropout_rate = Data['dropout_rate']
tokenizer = Data['tokenizer']
vocab = Data['vocab']

# Instantiate the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Define device
model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
model.load_state_dict(torch.load(output, map_location=device))
model.eval()

# Initialize Flask app
app = Flask(__name__)

# Define a route and a function that handles requests to it
@app.route('/', methods=['GET', 'POST'])

def index():
    # Home page
    if request.method == 'GET':
        return render_template('index.html', prompt='')
    
    # Page after user input
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        seq_len = int(request.form.get('seq'))
        temp = request.form.get('temperature')
        temperature = float(temp)
        seed = 69
        generation = generate(prompt, seq_len, temperature, model, tokenizer, 
                            vocab, device, seed)
        
        sentence = ' '.join(generation)
        
        if temp == "1.0" :
            fitting = "Superfitted"
        elif temp == "0.85" :
            fitting = "Slightly fitted"
        else :
            fitting = "Preplexed, more natural"
        
        return render_template('index.html', prompt=prompt, seq_len=seq_len, 
                               fitting=fitting, words=seq_len, sentence=sentence)

# Run the app if the script is executed directly
if __name__ == '__main__':
    app.run(debug=True)

# End of file