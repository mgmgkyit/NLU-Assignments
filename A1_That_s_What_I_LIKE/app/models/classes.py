import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
skg_args = pickle.load(open('skipgram.args', 'rb'))
word2index = skg_args.get('word2index', {})

# skipgram class module -----------------------------
class Skipgram(nn.Module):
    
    def __init__(self, voc_size, emb_size):
        super(Skipgram, self).__init__()
        self.embedding_center  = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)
    
    def forward(self, center, outside, all_vocabs):
        center_embedding     = self.embedding_center(center)  #(batch_size, 1, emb_size)
        outside_embedding    = self.embedding_center(outside) #(batch_size, 1, emb_size)
        all_vocabs_embedding = self.embedding_center(all_vocabs) #(batch_size, voc_size, emb_size)
        
        top_term = torch.exp(outside_embedding.bmm(center_embedding.transpose(1, 2)).squeeze(2))
        #batch_size, 1, emb_size) @ (batch_size, emb_size, 1) = (batch_size, 1, 1) = (batch_size, 1) 

        lower_term = all_vocabs_embedding.bmm(center_embedding.transpose(1, 2)).squeeze(2)
        #batch_size, voc_size, emb_size) @ (batch_size, emb_size, 1) = (batch_size, voc_size, 1) = (batch_size, voc_size) 
        
        lower_term_sum = torch.sum(torch.exp(lower_term), 1)  #(batch_size, 1)
        
        loss = -torch.mean(torch.log(top_term / lower_term_sum))  #scalar
        
        return loss
    
    def get_embed(self, index):
        # Retrieve the embedding for a specific word index.
        with torch.no_grad():  # Ensure gradients are not tracked
            return self.embedding_center(torch.LongTensor([index])).squeeze(0)

    def get_vector(self, word):
        id_tensor = torch.LongTensor([word2index[word]])
        id_tensor = id_tensor
        v_embed = self.embedding_center(id_tensor)
        u_embed = self.embedding_outside(id_tensor)
        word_embed = (v_embed + u_embed) / 2 

        return word_embed  
    
# -----------------------------

# SkipgramNeg class module -----------------------------
class SkipgramNeg(nn.Module):
    
    def __init__(self, voc_size, emb_size):
        super(SkipgramNeg, self).__init__()
        self.embedding_center  = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)
        self.logsigmoid        = nn.LogSigmoid()
    
    def forward(self, center, outside, negative):
        #center, outside:  (bs, 1)
        #negative       :  (bs, k)
        
        center_embed   = self.embedding_center(center) #(bs, 1, emb_size)
        outside_embed  = self.embedding_outside(outside) #(bs, 1, emb_size)
        negative_embed = self.embedding_outside(negative) #(bs, k, emb_size)
        
        uovc           = outside_embed.bmm(center_embed.transpose(1, 2)).squeeze(2) #(bs, 1)
        ukvc           = -negative_embed.bmm(center_embed.transpose(1, 2)).squeeze(2) #(bs, k)
        ukvc_sum       = torch.sum(ukvc, 1).reshape(-1, 1) #(bs, 1)
        
        loss           = self.logsigmoid(uovc) + self.logsigmoid(ukvc_sum)
        
        return -torch.mean(loss)
    
    def get_embed(self, index):
        # Retrieve the embedding for a specific word index.
        with torch.no_grad():  # Ensure gradients are not tracked
            device = self.embedding_center.weight.device  # Get the device of the embedding weights
            index_tensor = torch.LongTensor([index]).to(device)  # Move index to the same device
            return self.embedding_center(index_tensor).squeeze(0)

    def get_vector(self, word):
        id_tensor = torch.LongTensor([word2index[word]])
        id_tensor = id_tensor
        v_embed = self.embedding_center(id_tensor)
        u_embed = self.embedding_outside(id_tensor)
        word_embed = (v_embed + u_embed) / 2 

        return word_embed  

    def get_all_embeddings(self):
        # Retrieve all embeddings from the center embedding layer.
        with torch.no_grad():  # Ensure gradients are not tracked
            return self.embedding_center.weight

# -----------------------------

# Glove class module -----------------------------

class Glove(nn.Module):
    
    def __init__(self, voc_size, emb_size):
        super(Glove, self).__init__()
        self.center_embedding  = nn.Embedding(voc_size, emb_size)
        self.outside_embedding = nn.Embedding(voc_size, emb_size)
        
        self.center_bias       = nn.Embedding(voc_size, 1) 
        self.outside_bias      = nn.Embedding(voc_size, 1)
    
    def forward(self, center, outside, coocs, weighting):
        center_embeds  = self.center_embedding(center) #(batch_size, 1, emb_size)
        outside_embeds = self.outside_embedding(outside) #(batch_size, 1, emb_size)
        
        center_bias    = self.center_bias(center).squeeze(1)
        target_bias    = self.outside_bias(outside).squeeze(1)
        
        inner_product  = outside_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        #(batch_size, 1, emb_size) @ (batch_size, emb_size, 1) = (batch_size, 1, 1) = (batch_size, 1)
        
        loss = weighting * torch.pow(inner_product + center_bias + target_bias - coocs, 2)
        
        return torch.sum(loss)

    def get_embed(self, index):
        # Retrieve the embedding for a specific word index.
        with torch.no_grad():  # Ensure gradients are not tracked
            return self.center_embedding(torch.LongTensor([index])).squeeze(0)

    def get_vector(self, word):
        id_tensor = torch.LongTensor([word2index[word]])
        id_tensor = id_tensor
        v_embed = self.center_embedding(id_tensor)
        u_embed = self.outside_embedding(id_tensor)
        word_embed = (v_embed + u_embed) / 2 

        return word_embed  

# -----------------------------

