import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy



class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, out_channels//2, 1),
                                     nn.BatchNorm2d(out_channels//2),
                                     nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, 1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(),
            nn.Conv2d(out_channels//2, out_channels//2, 3, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, 1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(),
            nn.Conv2d(out_channels//2, out_channels//2, 5, padding=2),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels//2, 1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU()
        )
        self.reduce = nn.Conv2d(2*out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.reduce(torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1))


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #self.self_attn = DecayMultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        originaltext = tgt
        self_attn_result, self_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        resconnect1 = tgt + self.dropout1(self_attn_result)
        norm1 = self.norm1(resconnect1)
        cross_attn_result, cross_weights = self.multihead_attn(norm1, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        resconnect2 = norm1 + self.dropout2(cross_attn_result)
        norm2 = self.norm2(resconnect2)
        ffn_result = self.linear2(self.dropout(self.activation(self.linear1(norm2))))
        resconnect3 = norm2 + self.dropout3(ffn_result)
        norm3 = self.norm3(resconnect3)
        return norm3 + originaltext,self_weights, cross_weights
    
class StackTransformer(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        self_attn_weights_list = []
        cross_attn_weights_list = []
        for layer in self.layers:
            output, self_attn_weights, multihead_attn_weights = layer(output, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
            self_attn_weights_list.append(self_attn_weights)
            cross_attn_weights_list.append(multihead_attn_weights)
        if self.norm is not None:
            output = self.norm(output)
        return output, cross_attn_weights_list


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
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
        return self.dropout(x)


class FullTransformerDecoder(nn.Module):
    def __init__(self, encoder_dim, n_layers, feature_dim, vocab_size, n_head, max_lengths, word_vocab, dropout):
        super().__init__()

        self.feature_dim = feature_dim
        self.embedding_dim = feature_dim
        self.vocab_size = vocab_size
        self.max_length = max_lengths
        self.word_vocab = word_vocab
        self.dropout = nn.Dropout(p=dropout)
        self.conv1 = nn.Conv2d(encoder_dim * 2, feature_dim, kernel_size=1)
        self.resblock = Inception(feature_dim, feature_dim)
        self.vocab_embedding = nn.Embedding(vocab_size, feature_dim)
        decoder_layer = TransformerDecoderLayer(d_model=feature_dim, nhead=n_head, dim_feedforward=feature_dim * 4, dropout=dropout)
        self.transformer = StackTransformer(decoder_layer, n_layers)
        self.position_encoding = PositionalEncoder(feature_dim, max_len=max_lengths)
        self.fc = nn.Linear(feature_dim, vocab_size)
        self.cos = nn.CosineSimilarity(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.vocab_embedding.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x1, x2, encoded_captions, caption_lengths):
        #print(x1.shape)
        x_sam = self.cos(x1, x2)
        x = torch.cat([x1, x2], dim=1) + x_sam.unsqueeze(1) 
        x = self.resblock(self.conv1(x))
        batch, channel = x.size(0), x.size(1)
        x = x.view(batch, channel, -1).permute(2, 0, 1)

        word_length = encoded_captions.size(1)
        mask = torch.triu(torch.ones(word_length, word_length) * float('-inf'), diagonal=1).cuda()
        tgt_pad_mask = (encoded_captions == self.word_vocab['<NULL>']) | (encoded_captions == self.word_vocab['<END>'])
        word_embed = self.vocab_embedding(encoded_captions).transpose(1, 0)
        word_embed = self.position_encoding(word_embed)

        pred, cross_attn_weights = self.transformer(word_embed, x, tgt_mask=mask, tgt_key_padding_mask=tgt_pad_mask)
        pred = self.fc(pred)
        pred = pred.permute(1, 0, 2)

        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()

        return pred, encoded_captions, decode_lengths, sort_ind

    def sample(self, x1, x2):
        
        x_sam = self.cos(x1, x2)
        x = torch.cat([x1, x2], dim=1) + x_sam.unsqueeze(1) 

        x = self.resblock(self.conv1(x))
        batch, channel = x.size(0), x.size(1)
        x = x.view(batch, channel, -1).permute(2, 0, 1)  # HW, B, C

        tgt = torch.zeros(batch, self.max_length).to(torch.int64).cuda()

        mask = torch.triu(torch.ones(self.max_length, self.max_length) * float('-inf'), diagonal=1)
        mask = mask.cuda()
        tgt[:, 0] = torch.LongTensor([self.word_vocab['<START>']] * batch).cuda()  
        seqs = torch.LongTensor([[self.word_vocab['<START>']]] * batch).cuda()
        for step in range(self.max_length):
            tgt_pad_mask = (tgt == self.word_vocab['<NULL>'])
            word_emb = self.vocab_embedding(tgt)
            word_emb = word_emb.transpose(1, 0)  

            word_emb = self.position_encoding(word_emb)
            pred, cross_attn_weights = self.transformer(word_emb, x, tgt_mask=mask, tgt_key_padding_mask=tgt_pad_mask)

            pred = self.fc(self.dropout(pred))  
            scores = pred.permute(1, 0, 2) 
            scores = scores[:, step, :].squeeze(1)  
            predicted_id = torch.argmax(scores, axis=-1)
            seqs = torch.cat([seqs, predicted_id.unsqueeze(1)], dim=-1)
            if predicted_id == self.word_vocab['<END>']:
                break
            if step < (self.max_length - 1):  
                tgt[:, step + 1] = predicted_id
        seqs = seqs.squeeze(0)
        seqs = seqs.tolist()
        
        return seqs, cross_attn_weights 
