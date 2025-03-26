import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv, GINConv, GATConv, global_add_pool, Set2Set

class FeatureSelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(FeatureSelfAttention, self).__init__()
        self.feature_dim = feature_dim

        self.wq = nn.Linear(feature_dim, feature_dim, bias=False)
        self.wk = nn.Linear(feature_dim, feature_dim, bias=False)
        self.wv = nn.Linear(feature_dim, feature_dim, bias=False)

        self.fc_out = nn.Linear(feature_dim, feature_dim)  # 输出层

    def forward(self, x, softmax=False):
        batch_size, feature_dim = x.shape 

        q = self.wq(x)  
        k = self.wk(x)  
        v = self.wv(x)  

        attn_scores = torch.matmul(q.unsqueeze(2), k.unsqueeze(1)) / (feature_dim ** 0.5)
        if softmax:
            attn = torch.softmax(attn_scores, dim=-1)  

        out = torch.matmul(attn_scores, v.unsqueeze(2)).squeeze(2)  
        out = self.fc_out(out)  

        return out, attn_scores  


class TransformerModel(nn.Module):
    def __init__(self, num_layers, feature_size, d_k, num_heads, max_relative_distance=10, dropout_rate=0.2, enhance=1, activation_function=nn.GELU):
        super(TransformerModel, self).__init__()
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(feature_size, d_k, num_heads, max_relative_distance, dropout_rate, enhance, activation_function)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask_pad, position_idx):
        attention_scores = []
        for block in self.transformer_blocks:
            x, scores = block(x, mask_pad, position_idx)
            attention_scores.append(scores)

        avg_attention_scores = torch.mean(torch.stack(attention_scores, dim=0), dim=0)
        return x, avg_attention_scores
    
class TransformerBlock(nn.Module):
    def __init__(self, feature_size, d_k, num_heads, max_relative_distance=10, dropout_rate=0.2, enhance=1, activation_function=nn.GELU):
        super(TransformerBlock, self).__init__()
        self.self_attention = MultiHeadSelfAttention(feature_size, d_k, num_heads, max_relative_distance, enhance, dropout_rate)
        self.feed_forward = FeedForwardNetwork(feature_size, dropout_rate, activation_function)

    def forward(self, x, mask_pad, position_idx):
        attention_output, attention_weights = self.self_attention(x, x, x, mask_pad, position_idx)
        output = self.feed_forward(attention_output)
        return output, attention_weights

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feature_size, d_k, num_heads, max_relative_distance=7, enhance=1, dropout_rate=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_k = d_k
        self.num_heads = num_heads
        self.query_layer = nn.Linear(feature_size, enhance * self.d_k * self.num_heads)
        self.key_layer = nn.Linear(feature_size, enhance * self.d_k * self.num_heads)
        self.value_layer = nn.Linear(feature_size, self.d_k * self.num_heads)
        self.output_layer = nn.Linear(self.d_k * self.num_heads, feature_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        if max_relative_distance > 0:
            self.relative_position_embedding_k = nn.Embedding(2 * max_relative_distance + 1, self.num_heads)
            self.relative_position_embedding_b = nn.Embedding(2 * max_relative_distance + 1, self.num_heads)
        
        self.max_relative_distance = max_relative_distance
        self.enhance = enhance

    def forward(self, queries, keys, values, mask_pad=None, position_idx=None):
        batch_size, seq_length, _ = queries.shape
        
        queries = self.query_layer(queries).reshape(batch_size, seq_length, self.num_heads, self.d_k * self.enhance).transpose(1, 2)
        keys = self.key_layer(keys).reshape(batch_size, seq_length, self.num_heads, self.d_k * self.enhance).transpose(1, 2)
        values = self.value_layer(values).reshape(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(queries, keys.transpose(-1, -2)) / np.sqrt(self.d_k)

        if self.max_relative_distance > 0:
            if position_idx is None:
                relative_position_table = torch.abs(torch.arange(0, seq_length).reshape(1, -1, 1) - torch.arange(0, seq_length).reshape(1, 1, -1)).float()
            else:
                relative_position_table = torch.abs(position_idx.reshape(batch_size, seq_length, 1) - position_idx.reshape(batch_size, 1, seq_length)).float()
            
            relative_position_table[relative_position_table > self.max_relative_distance] = self.max_relative_distance + torch.log2(relative_position_table[relative_position_table > self.max_relative_distance] - self.max_relative_distance).float()
            relative_position_table = torch.clip(relative_position_table, min=0, max=self.max_relative_distance * 2).long().to(queries.device)
            
            scores = scores * self.relative_position_embedding_k(relative_position_table).transpose(1, -1).reshape(-1, self.num_heads, seq_length, seq_length) + self.relative_position_embedding_b(relative_position_table).transpose(1, -1).reshape(-1, self.num_heads, seq_length, seq_length)

        if mask_pad is not None:
            scores = scores.masked_fill((mask_pad == 0).unsqueeze(dim=1), -2**32 + 1)

        attention_weights = self.dropout(F.softmax(scores, dim=3))

        output = torch.matmul(attention_weights, values)
        output = output.transpose(1, 2).reshape(batch_size, seq_length, -1)

        output = self.output_layer(output)
        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    def __init__(self, feature_size, dropout_rate=0.2, activation_function=nn.GELU):
        super(FeedForwardNetwork, self).__init__()
        self.layer_norm1 = nn.LayerNorm([feature_size])
        self.layer_norm2 = nn.LayerNorm([feature_size])
        self.ffn = nn.Sequential(
            nn.Linear(feature_size, feature_size * 4),
            activation_function(),
            nn.Linear(feature_size * 4, feature_size)
        )
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.layer_norm1(x)
        ffn_output = self.ffn(x)
        return self.layer_norm2(ffn_output)


class Channel_Embedding(nn.Module):
    def __init__(self, classNum, tokenNum, embedding_dim=128, enhance=1, layer_num=4, hidden_dim=256, head_num=4, max_relative_distance=20, embDropout=0.2, paddingIdx=0):
        super(Channel_Embedding, self).__init__()
        self.embedding =  nn.Embedding(tokenNum, embedding_dim, padding_idx=paddingIdx)
        self.dropout = nn.Dropout(p=embDropout)
        
        self.backbone = TransformerModel(layer_num, feature_size=embedding_dim, d_k=hidden_dim//head_num, num_heads=head_num, max_relative_distance=max_relative_distance, dropout_rate=0.1, enhance=enhance)


    def forward(self, data):
        # print(data['sub_seq_embedding'].device, self.embedding.weight.device)
        x = data['sub_seq_embedding'] @ self.embedding.weight 
        x = self.dropout(x)
        
        x,attenton_socres = self.backbone(x, None, None) 
        x_mean = torch.mean(x, dim=1) 
        return x_mean, attenton_socres
        


class LncTracker(nn.Module):
    
    def __init__(self, tokenizer,n_features, hidden_dim, embedding_dim, n_classes, n_conv_layers=3, dropout=0.2,
                 conv_type="GAT", n_trans_layers=8, head_num=8, softmax=False,
                 batch_norm=True,  batch_size=128):
        super(LncTracker, self).__init__()

        #
        self.batch_size = batch_size
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        '''Transformer'''
        self.transformer = Channel_Embedding(tokenizer.label_count, tokenNum=tokenizer.token_count,embedding_dim=embedding_dim, enhance=1,
                              layer_num=n_trans_layers, hidden_dim=256, head_num=head_num, max_relative_distance=20,
                              embDropout=dropout, paddingIdx=tokenizer.token2id['[PAD]'])

        '''GNN1'''
        self.convs.append(self.get_conv_layer(n_features, hidden_dim*2, conv_type=conv_type))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim*2))

        for i in range(n_conv_layers - 1):
            self.convs.append(self.get_conv_layer(hidden_dim*2, hidden_dim*2, conv_type=conv_type))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim*2))
   
        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.mlp_cksnap = nn.Sequential(nn.Linear(96, 128, bias=True), nn.ReLU())
        self.mlp_kmer = nn.Sequential(nn.Linear(1024, 128, bias=True), nn.ReLU())
        
        self.self_attention1 = FeatureSelfAttention(feature_dim=96)
        self.self_attention2 = FeatureSelfAttention(feature_dim=1024)

        self.fc1 = nn.Linear(512,128)#
        self.fc2 = nn.Linear(128, n_classes)
    

        self.pooling1 = Set2Set(hidden_dim*2, processing_steps=2)

        self.dropout = nn.Dropout(dropout)
        self.conv_type = conv_type
        self.batch_norm = batch_norm


    def forward(self, data): 
        # print("calculation start...")

        g1, adj1, edge_attr1, batch1 = data.x, data.edge_index, data.edge_attr, data.batch 

        x_transformer, transformer_as = self.transformer(data)

        x_cksnap=data.cksnap 
        x_kmer=data.kmer 

        for i, con in enumerate(self.convs):
            g1 = self.apply_conv_layer(con, g1, adj1, edge_attr1, conv_type=self.conv_type) 
            g1 = self.batch_norms[i](g1) if self.batch_norm else g1
            g1 = self.dropout(g1) 

        g1 = self.pooling1(g1, batch1) 

        g1 = self.dropout(g1) 
 

        ss=self.batch_size

        x1, weight_cksnap = self.self_attention1(x_cksnap) 
        x1 = self.mlp_cksnap(x1)
        x2, weight_kmer = self.self_attention2(x_kmer)
        x2 = self.mlp_kmer(x2)

        x = torch.cat((x1, x2, x_transformer, g1),dim=1)
        embeddings = x
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        
        output = torch.sigmoid(x)
        return embeddings, output


    @staticmethod
    def get_conv_layer(n_input_features, n_output_features, conv_type="GCN"):
        if conv_type == "GCN":
            return GCNConv(n_input_features, n_output_features)
        elif conv_type == "GAT":
            return GATConv(n_input_features, n_output_features)
        elif conv_type == "MPNN":
            net = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, n_input_features *
                                                                      n_output_features))
            return NNConv(n_input_features, n_output_features, net)
        elif conv_type == "GIN":
            net = nn.Sequential(nn.Linear(n_input_features, n_output_features), nn.ReLU(),
                                nn.Linear(n_output_features, n_output_features))
            return GINConv(net)
        else:
            raise Exception("{} convolutional layer is not supported.".format(conv_type))

    @staticmethod
    def apply_conv_layer(conv, x, adj, edge_attr, conv_type="GCN"):
        if conv_type in ["GCN", "GAT", "GIN"]:
            return conv(x, adj)
        elif conv_type in ["MPNN"]:
            return conv(x, adj, edge_attr)
        else:
            raise Exception("{} convolutional layer is not supported.".format(conv_type))