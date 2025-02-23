import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the default directory to the current directory ***
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('model/bert.param', 'rb') as f:
    bert_param = pickle.load(f)
    word2id = bert_param['word2id']

# embedding
# I will use BERT Model Definition from previous notebook
class Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, n_segments, d_model, device):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).to(self.device).unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

class BERT(nn.Module):
    def __init__(self, n_layers, n_heads, d_model, d_ff, d_k, 
                 n_segments, vocab_size, max_len, device):
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size, max_len, n_segments, d_model, device)
        self.layers = nn.ModuleList([EncoderLayer(n_heads, d_model, d_ff, d_k, device) 
                                     for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))
        self.device = device

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        return output

    # New helper for sentence encoding (for classification, similar to S-BERT)
    def get_last_hidden_state(self, input_ids, attention_mask):
        # Create dummy segment_ids (all zeros)
        segment_ids = torch.zeros_like(input_ids).to(self.device)
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for layer in self.layers:
            output, _ = layer(output, enc_self_attn_mask)
        return output


def tokenizer(sentences, max_length, padding='max_length', truncation=True):
    tokenized_outputs = {"input_ids": [], "attention_mask": []}
    for sentence in sentences:
        tokens = sentence.lower().split()
        token_ids = [word2id.get(token, word2id['[UNK]']) for token in tokens]
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        attention_mask = [1] * len(token_ids)
        if padding == 'max_length':
            padding_length = max_length - len(token_ids)
            token_ids += [word2id['[PAD]']] * padding_length
            attention_mask += [0] * padding_length
        tokenized_outputs["input_ids"].append(token_ids)
        tokenized_outputs["attention_mask"].append(attention_mask)
    return tokenized_outputs

def get_attn_pad_mask(seq_q, seq_k, device):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1).to(device)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_k, device):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_heads, d_model, d_k, device)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, device):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_k * n_heads)
        self.device = device

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention(self.d_k, self.device)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        output = nn.Linear(self.n_heads * self.d_k, self.d_model).to(self.device)(context)
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual), attn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, device):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.FloatTensor([d_k])).to(device)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool

def configurations(u,v):
    # build the |u-v| tensor
    uv = torch.sub(u, v)   # batch_size,hidden_dim
    uv_abs = torch.abs(uv) # batch_size,hidden_dim
    
    # concatenate u, v, |u-v|
    x = torch.cat([u, v, uv_abs], dim=-1) # batch_size, 3*hidden_dim
    return x

# cosine similarity function
def cosine_similarity(u, v):
    u = u.flatten()  # Convert (1, 768) → (768,)
    v = v.flatten()  # Convert (1, 768) → (768,)

    dot_product = np.dot(u, v) 
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    similarity = dot_product / (norm_u * norm_v)
    return similarity

def calculate_similarity(model, tokenizer, sentence_a, sentence_b, device, max_length=128):
    # Use the custom tokenizer which expects a list of sentences.
    inputs_a = tokenizer([sentence_a], max_length=max_length, padding='max_length', truncation=True)
    inputs_b = tokenizer([sentence_b], max_length=max_length, padding='max_length', truncation=True)
    
    # Convert lists to torch tensors and move to device.
    input_ids_a = torch.tensor(inputs_a['input_ids']).to(device)
    attention_a = torch.tensor(inputs_a['attention_mask']).to(device)   
    input_ids_b = torch.tensor(inputs_b['input_ids']).to(device)
    attention_b = torch.tensor(inputs_b['attention_mask']).to(device)
    
    # Use the model's helper method to get token embeddings.
    u = model.get_last_hidden_state(input_ids_a, attention_a)  # shape: [1, seq_len, hidden_dim]
    v = model.get_last_hidden_state(input_ids_b, attention_b)  # shape: [1, seq_len, hidden_dim]
    
    # Get mean pooled sentence embeddings (shape: [1, hidden_dim])
    u_mean = mean_pool(u, attention_a).detach().cpu().numpy()
    v_mean = mean_pool(v, attention_b).detach().cpu().numpy()
    
    # Calculate cosine similarity (result is a scalar)
    similarity_score = cosine_similarity(u_mean, v_mean)
    return similarity_score


