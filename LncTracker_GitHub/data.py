import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
import torch,random,os
from torch_geometric.data import Data, InMemoryDataset, Dataset, Batch
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from itertools import permutations
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
import logging,pickle
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class SequenceTokenizer:
    def __init__(self, sequences, labels, k=3):
        print('Tokenizing the data...')

        # Padding sequences with '-'
        padded_sequences = [('-' * (k // 2)) + seq + ('-' * (k // 2)) for seq in sequences]
        
        # Sub-sequences generation
        sub_sequences = [
            [seq[j - k // 2:j + k // 2 + 1] for j in range(k // 2, len(seq) - k // 2)]
            for seq in padded_sequences
        ]
        
        # Token-to-ID and ID-to-Token dictionaries
        token_count = 3
        id2token = ["[MASK]", "[PAD]", "[CLS]"]
        token2id = {"[MASK]": 0, "[PAD]": 1, "[CLS]": 2}
        
        # Update token dictionaries with sub-sequences
        for sub_seq in tqdm(sub_sequences):
            for token in sub_seq:
                if token not in token2id:
                    token2id[token] = token_count
                    id2token.append(token)
                    token_count += 1
        
        # Save token dictionaries
        self.id2token = id2token
        self.token2id = token2id
        self.token_count = token_count

        # Label-to-ID and ID-to-Label dictionaries
        label_count = 0
        id2label = []
        label2id = {}
        for label_list in labels:
            for label in label_list:
                if label not in label2id:
                    label2id[label] = label_count
                    id2label.append(label)
                    label_count += 1

        # Save label dictionaries
        self.id2label = id2label
        self.label2id = label2id
        self.label_count = label_count

        # MultiLabelBinarizer for labels
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(labels)

        # OneHotEncoder for tokens
        self.ohe = OneHotEncoder()
        self.ohe.fit([[i] for i in range(self.token_count)])

    def tokenize_sub_sequences(self, sequence, embedding_dim):
        # Convert tokens in the sequence to their corresponding embeddings
        token_embeddings = F.adaptive_avg_pool1d(
            torch.tensor(
                self.ohe.transform([
                    [self.token2id.get(token, self.token2id["[MASK]"])]
                    for token in sequence
                ]).toarray(),
                dtype=torch.float32
            ).transpose(-1, -2),
            output_size=embedding_dim
        ).transpose(-1, -2).unsqueeze(0)

        return token_embeddings  # Shape: (1, embedding_dim, num_tokens)
    
    def save_tokenizer(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

def load_tokenizer(filename):
    with open(filename, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


def sinusoidal_position_encoding(num_nodes, pos_encoding_dim):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / pos_encoding_dim) for j in range(pos_encoding_dim)]
        for pos in range(num_nodes)
    ])
    # print("one position_enc!")
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2]) 
    return torch.tensor(position_enc, dtype=torch.float)

word_to_ix = {"X": [0,0,0,0,0], "A": [0,1,0,0,0], "G": [0,0,1,0,0], "C": [0,0,0,1,0], "T": [0,0,0,0,1], "U": [0,0,0,0,1]}
N_to_EIIP = {"<PAD>": 0, "A": 0.1260, "G": 0.0806, "C": 0.1340, "T": 0.1335, "U": 0.1335}
N_to_NCP = {"<PAD>": [0,0,0], "A": [1,1,1], "G": [1,0,0], "C": [0,1,0], "T": [0,0,1], "U": [0,0,1]}

def prepare_sequence(seq,to_ix, pos_dim=10):
    idxs=[]
    pos_encode = sinusoidal_position_encoding(len(seq), pos_dim)
    for j,char in enumerate(seq):
        ANF=[seq[0:j+1].count(seq[j])/(j+1)] # 累积核苷酸频率
        subidx=to_ix[char]+[N_to_EIIP[char]]+N_to_NCP[char]+ANF # 5+1+3+1+10=20维特征
        idxs.append(subidx)
    idx = torch.tensor(idxs, dtype=torch.float)
    return torch.concat([idx,pos_encode],axis=1)

def dotbracket_to_graph(dotbracket):
    G = nx.Graph()
    bases = []

    for i, c in enumerate(dotbracket):
        if c == '(':
            bases.append(i)
        elif c == ')':
            neighbor = bases.pop()
            G.add_edge(i, neighbor, edge_type='base_pair') # 将配对核苷酸连接
        elif c == '.':
            G.add_node(i)
        else:
            print("Input is not in dot-bracket notation!")
            return None

        if i > 0:
            G.add_edge(i, i - 1, edge_type='adjacent') # 按序列顺序将相邻核苷酸连接
    return G

class UnitLncRNADataset(InMemoryDataset): # folding1表征的二级结构
    def __init__(self, root='data', dataset='g1', view='train', 
                 df_data=None, tokenizer=None, foldings=None, fea_kmer=None, fea_cksnap=None, emb_k=3, emb_dim=512, transform=None,
                 pre_transform=None, device="cuda"
                 ):

        #root is required for save preprocessed data, default is '/tmp'
        super(UnitLncRNADataset, self).__init__(root, transform, pre_transform)
        # self.records = records
        self.dataset = dataset
        self.view=view
        self.foldings = foldings
        self.fea_kmer = fea_kmer
        self.fea_cksnap = fea_cksnap
        self.emb_k = emb_k
        self.emb_dim = emb_dim
        self.tokenizer = tokenizer
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0], map_location=device)
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(df_data)
            self.data, self.slices = torch.load(self.processed_paths[0], map_location=device)
        
    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset+'_'+ self.view+ '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, df):
        data_list = []
        for row in df.itertuples():
            des = str(row.Description)
            seq_str = str(row.Sequence)
            if seq_str not in self.foldings:
                print("pass 1 records....")
                continue
            dot_bracket_string_1 = self.foldings[seq_str][0]
            # dot_bracket_string_2 = foldings_2[seq_str][0]
            seq_attr = prepare_sequence(seq_str, word_to_ix)  # each seq dim:(len(seq),10)

            # 将位置信息进行one-hot编码
            locations = str(row.Label).split(',')
            label_embedded = self.tokenizer.mlb.transform([locations])

            graph = dotbracket_to_graph(dot_bracket_string_1)

            x = torch.tensor(seq_attr)
            y = torch.Tensor(label_embedded)
            y = y.view(1, len(self.tokenizer.mlb.classes_))


            edges = list(graph.edges(data=True)) # data=True时查看边的所有属性，默认为边的俩顶点
            edge_attr = torch.Tensor([[0, 1] if e[2]['edge_type'] == 'adjacent' else [1, 0] for e in edges])
            edge_index = torch.LongTensor(list(graph.edges())).t().contiguous()

            # data.cksnap和data.kmer表示序列特征
            # data.x表示二级结构中的节点特征，data.edge_index表示所有边连接的两个节点，data.edge_attr表示边的类型（邻接和配对）
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.cksnap = torch.tensor(self.fea_cksnap[seq_str],dtype=torch.float32)
            data.kmer = torch.tensor(self.fea_kmer[seq_str],dtype=torch.float32)
            data.label = str(row.Label).split(',')
            data.sLen = len(seq_str)
            data.rowseq = seq_str
            data.des = des
            # data.tokenizedSeqArr,data.maskPAD = self.tokenizer.tokenize_sentences([seq_str], train=True)
            tmp = '-'*(self.emb_k//2)+seq_str+'-'*(self.emb_k//2) 
            data.subseqs = [tmp[j-self.emb_k//2:j+self.emb_k//2+1] for j in range(self.emb_k//2,len(tmp)-self.emb_k//2)]
            data.sub_seq_embedding = self.tokenizer.tokenize_sub_sequences(data.subseqs, self.emb_dim)

            data_list.append(data)
            # print("1 ros done!")
        
        data, slices = self.collate(data_list) 
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

def unit_collate_func(batch):
    batch_data = Batch.from_data_list(batch)
    batch_data.y = torch.cat([data.y for data in batch], dim=0)
    batch_data.cksnap = torch.stack([data.cksnap for data in batch], dim=0)
    batch_data.kmer = torch.stack([data.kmer for data in batch], dim=0)
    batch_data.seqLens = [data.sLen for data in batch]
    batch_data.sub_seq_embedding = torch.cat([i['sub_seq_embedding'] for i in batch], dim=0)
    batch_data.des = [data.des for data in batch]
    return batch_data

