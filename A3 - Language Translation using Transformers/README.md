# Assignment A3 : Language Translator using Transformers
# Here, I've experimented langage translation with Transformers
Student ID: st125214

# About this task
The intention of this task is to experiment on Transformers for langage translation.

In the training the layers are arranged as below : 
Encoder Layer -> Multihead Attention Layer -> Position-wise Feedforward Layer -> Decoder Layer are used in this technique.

Three different attention mechanisms were experimented in this task, and the most effective attention mechanism was used to create a web application which tries to translate input language into another.

For language translation transformers, huge amounts of parallel dual-language datasets were needed, along with the intensive GPU resources (GPU memory, parallel tensor cores, GPU time).

My experiment uses a limited size of parallel corpora, and was run on the single GPU with limited GPU RAM, and training parameters reduced, and therefore the results cannot be precise. It will be just to experiment and demonstrate the concept, not to be used in production.

For practical use, we need to use larger text dataset, increase the training parameters, and deploy to use a decent GPU (multi-GPU parallel computing) with enough GPU RAM to get decent results.

===========================================
## Task 1 : Language Pair (Parallel Corpora)
===========================================
## Source Material
1. In my experiment, I use language pair created by Aung K. Htet.
   His language pair contains 7,500 human-translated sentence pairs, and contains 392,702 records.
2. Data source: https://huggingface.co/datasets/akhtet/myanmar-xnli
   Github Link : https://github.com/akhtet/myXNLI
   
## Preparing the source material
1. Since the original dataset has Label, Genre, and two different translated sentences, I have to reduce the dataset into one-third of the original size, and use only part of the set due to limitations on GPU resource allocations.
2. Tokenization of english language is easy because of availability of different tokenizers.
But tokenization of Myanmar text is quite challenging task in here, because myanmar text processing is still under development process, and very limited MM text tokenizers were available.
I have to create a custom MM text tokenizer from PyICU Unicode Library.

================================================
## Task 2 : Experiment with Attention Mechanisms
================================================ 
1. Result of different Attention Mechanisms

| Attentions | Training Loss | Training PPL | Validation Loss | Validation PPL | Total Training Time |
|------------|---------------|--------------|-----------------|----------------|---------------------|
| General Attention        | 4.105 | 60.638 | 4.070 | 58.546 | 2m 15s  |
| Additive Attention       | 2.977 | 19.626 | 3.244 | 25.633 | 3m 29s  |
| Multiplicative Attention | 3.473 | 32.245 | 3.574 | 35.673 | 2m 23s  |
 
=======================================
## Task 3 : Evaluation and Verification
=======================================

1. Among the three attantion mechanisms, Additive Attention results as the most effective.

2. Performance Plots for all models are as below :
![Training Comparison](<performance plots.png>)

3. Attention Maps
- Attention Map for General Attention is as below :
![General Attention](<General Attention map.png>)

- Attention Maps for Additive Attention is as below :
![Additive Attention](<Additive Attention map.png>)

- Attention Map for Multiplicative Attention is as below :
![Multiplicative Attention](<Multiplicative Attention map.png>)

4. Results analysis
- Although Additive Attention has best results, it is the lowest in computation efficiency, since it takes 1.5 times more than other mechanisms for training.
- From the perplexity point of view, additive attention is best-scoring.
- Therefore, overall, Additive Attention mechanism is the most effective among the three mechanisms.
- Since this experimentation's main weakness is the lack of GPU resources and training time, the results may change with larger parallel corpus, and different training metrics.  

==================================================
## Task 4 : Web Application - Language Translation
==================================================
## Application Development
Application was developed with flask and python, with html css template as frontend.

## How to run the web app
from the command prompt, run : "python app/app.py"
Access the app from http://127.0.0.1:5000

## Result Screenshots
Sample results screenshots were also placed under the folder 'screens'.

