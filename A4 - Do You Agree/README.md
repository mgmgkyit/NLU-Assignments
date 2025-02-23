# Assignment A4 : Do you agree
# Here, I've experimented a BERT, with scratch and pre-trainining.
Student ID: st125214

# About this task
The intention of this task is to experiment BERT, both from scratch, and using pretrained models.

We can understand more on Natural Language Inference using BERT model's ability to understand semantic relationships between sentences.

# Limitations of the task
To train the BERT model to get a relative decent results, we need to train using training data with more than billions of sentences.
This requires multiple GPUs, enough GPU RAM, and training time.
Our experiment has limitations on those, therefore, the BERT results are nowhere near satisfactory.

My experiment will use a limited size of sentences, only around 100k sentences.


======================================
## Task 1 : Training BERT from Scratch
======================================
## Source Material
1. Data Source : https://huggingface.co/datasets/agentlans/high-quality-english-sentences

2. Dataset Description: 
This dataset contains a collection of high-quality English sentences sourced from C4 and FineWeb (not FineWeb-Edu). The sentences have been carefully filtered and processed to ensure quality and uniqueness.
"High-quality" means they're legible English and not spam, although they may still have spelling and grammar errors.
   
## Preparing the source material
1. This dataset contains around 1.7 million lines, therefore I use only 100k lines from the whole dataset.

2. Sentences are changed to all lower-case, and special characters were removed from the sentences later.

3. After training, trained model weights were saved for later tasks.

=================================================
## Task 2 : Sentence Embedding with Sentence BERT
================================================= 
1. Dataset : I only use SNLI dataset to reduce memory requirements
Here, I implement trained BERT from task 1 with siamese network structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity.

2. Average Cosine Similarity [S-Bert]: 0.9926
 
===================================
## Task 3 : Evaluation and Analysis
===================================

1. Model Evaluation

| Model Tyoe      | Accuracy | Precision | Recall | F1 Score |
|-----------------|----------|-----------|--------|----------|
| My Model        | 0.3400| 0.1156 | 0.3400| 58.546 | 0.1725  |
|                 |       |        |       |        |         |


2. Analysis : Challenges and Limitations
- Computational Resources : Training BERT requires significant computational power and memory. To be able to properly train a BERT model, an industrial setup of GPU array is needed, with enough training time, which we cannot perform in our experiment.

- Data Quality and Size : Size of training data and quality also have effects on bert model. Since my model use only 100k lines, the results were not near satisfactory.

- Training Complexity : Since BERT training involves many parameters, with multiple layers and steps, training requires multiple experimentation, with parameter tuning and results observation. This requires time and computational resources also.

3. Training Parameters
The following training parameters were used for training :
    max_len = 1000
    n_layers = 6
    n_heads = 8
    d_model = 768
    d_ff = d_model * 4
    d_k = d_v = 64
    n_segments = 2

Even with batch size of 32 is used, but the results are not satisafctory, possibly due to limited training dataset size.

Increasing the number of layers <n_layers> may have improved the model performance, but the time and GPU/RAM resources demand is also increased.

Num_epochs : 3 epochs were used, as during experimentation with different epochs, after 3 epochs, accruacy becomes stable.


=============================================
## Task 4 : Web Application - Text Similarity
=============================================
## Application Development
Application was developed with flask and python, with html css template as frontend.

## How to run the web app
from the command prompt, run : "python app/app.py"
Access the app from http://127.0.0.1:5000

## Result Screenshots
Sample results screenshots were also placed under the folder 'screens'.

(<Screenshot 2025-02-23 165424.png>)
(<Screenshot 2025-02-23 165453.png>)