import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

batch_size = 64


class SelfAttention(nn.Module):
    def __init__(self,dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))  #mask==0表示被屏蔽
        attention = self.softmax(scores)
        attention = self.dropout(attention)
        output = torch.matmul(attention, V)
        return output, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model              #512
        self.num_heads = num_heads          #8
        self.d_k = d_model // num_heads     #64
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.linear_q = nn.Linear(d_model, d_model) #后续再进行拆分
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model) #多头拼接后的输出
        self.attention = SelfAttention(dropout)
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        Q = self.linear_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.linear_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.linear_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        out, attention = self.attention(Q, K, V, mask=mask)
        #contiguous()函数将tensor变成在内存中连续分布的形式，可以加快速度,防止view出错
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.linear_out(out)
        out = self.dropout(out)
        out = self.norm(out + q)  #这里加上Q的原因
        return out, attention

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  #2048
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        x = self.norm(x + self.linear2(self.dropout(F.relu(self.linear1(x)))))
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        x=self.self_attn(x,x,x,mask=mask)[0]
        x=self.ffn(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)  #带mask的
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, tgt, memory,tgt_mask=None,memory_mask=None):
        # src原序列,tgt目标序列,memory编码后的序列
        x =self.self_attn(tgt,tgt,tgt,mask=tgt_mask)[0]  #_代表忽略
        x =self.cross_attn(x,memory,memory,mask=memory_mask)[0]
        x =self.ffn(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):  #max_len是最大长度
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class Encoder(nn.Module):
    def __init__(self,vocab_size, d_model, num_heads, d_ff, num_layers=6, dropout=0.1,max_len=5000):  #vocab_size是词表大小
        super().__init__()
        self.embedding_dim = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])  #堆叠结构
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.embedding(x)*math.sqrt(self.embedding_dim)
        x = self.pos_embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self,vocab_size, d_model, num_heads, d_ff, num_layers=6, dropout=0.1,max_len=5000):
        super().__init__()
        self.embedding_dim = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self,tgt,memory,tgt_mask=None,memory_mask=None):
        x = self.embedding(tgt)*math.sqrt(self.embedding_dim)
        x = self.pos_embedding(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.fc(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, dropout=0.1,max_len=5000):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout,max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout,max_len)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, src_mask, tgt_mask)
        return output

src_vocab_size = 10000
tgt_vocab_size = 10000
model = Transformer(src_vocab_size, tgt_vocab_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)






