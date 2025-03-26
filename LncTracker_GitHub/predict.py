import argparse
import pandas as pd
import numpy as np
import pickle
import csv
import os
import sys
from model import LncTracker
from data import UnitLncRNADataset, unit_collate_func
from torch.utils.data import DataLoader
from metrics import *
import utils
from utils import *
from data import *
from train import predict
from Bio import SeqIO
import torch
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default="test", help='input fasta')
parser.add_argument('--output_path', default="test", help='output path')
parser.add_argument('--device', default="cpu", help='cpu or cuda')
args = parser.parse_args()

device = args.device
input_path = args.input_path
output_path = args.output_path
batch_size = 8
learningrate = 0.0001
epochs = 200
patience = 20
# device = "cuda"
thres = 0.5
checkpoint_folder = "./checkpoints"

# device = "cuda"
if __name__ == '__main__':
    sequence_path = input_path
    folding_path = './features/linearfold_ncbi_unique.pkl'
    cksnap_path = './features/lnc_5_cksnap_ncbi_unique.pkl'
    kmer_path = './features/lnc_5_kmer_ncbi_unique.pkl'

    testDF = utils.read_fasta_to_df(sequence_path)

    # out_fasta_dot = "temp_fasata_dot"
    # utils.linear_fold(list(testDF['Sequence']), list(testDF['Description']), out_fasta_dot)
    # utils.dot_fasta_to_pkl(out_fasta_dot+".fasta", folding_path)

    # kw = {'order': 'ACGT'}
    # cksnap_encoding = utils.CKSNAP(sequence_path, gap=5, **kw)
    # kmer_encoding = utils.Kmer(sequence_path, k=5, type="RNA", upto=False, normalize=True, **kw)
    # with open(cksnap_path, 'wb') as handle:
    #     pickle.dump(cksnap_encoding, handle)
    # with open(kmer_path, 'wb') as handle:
    #     pickle.dump(kmer_encoding, handle)

    foldings = pickle.load(open(folding_path, "rb"))
    features_cksnap = pickle.load(open(cksnap_path, "rb"))
    features_kmer = pickle.load(open(kmer_path, "rb"))

    all_locations = set()
    for i in testDF['Label']:
        all_locations.update(i.split(','))
    all_locations = sorted(list(all_locations))

    tokenizer = load_tokenizer("tokenizer.pkl")

    test_dataset = UnitLncRNADataset(root='./data_prepared', dataset="predict",view=f'test',df_data=testDF, tokenizer=tokenizer, foldings=foldings, fea_kmer=features_kmer, fea_cksnap=features_cksnap, emb_k=3, emb_dim=512, device=device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=unit_collate_func)

    model = LncTracker(tokenizer=tokenizer,n_features=20, hidden_dim=32, embedding_dim=128, n_classes=tokenizer.label_count,
                            n_conv_layers=4, conv_type="GAT", n_trans_layers=8, head_num=8, 
                            dropout=0.2, batch_norm=True,
                            batch_size=batch_size)
    model.load_state_dict(torch.load(os.path.join(checkpoint_folder, "model_lncTracker.pth"), map_location=device))
    model.to(device)
    res = predict(model, test_loader, device=device)

    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Seq_ID"] + list(tokenizer.mlb.classes_))
        for row in res:
            writer.writerow(row)
