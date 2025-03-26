import pandas as pd
import numpy as np
import pickle
import os
import sys
from model import LncTracker
from data import Tokenizer, UnitLncRNADataset, unit_collate_func
from torch.utils.data import DataLoader
from metrics import *
import utils
from utils import *
from data import *
from train import train_valid, valid_step
from Bio import SeqIO
import torch
import torch.optim as optim

if __name__ == '__main__':
    device = "cuda"

    sequence_path = '/home/zhangyuchen/haoyi/Project/lncRNA/data/RNAlocatev3/lncRNA_ncbi_unique_6000.fa'
    folding_path = './features/linearfold_ncbi_unique.pkl'
    cksnap_path = './features/lnc_5_cksnap_ncbi_unique.pkl'
    kmer_path = './features/lnc_5_kmer_ncbi_unique.pkl'

    df = utils.read_fasta_to_df(sequence_path)

    # out_fasta_dot = "temp_fasata_dot"
    # utils.linear_fold(list(df['Sequence']), list(df['Description']), out_fasta_dot)
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
    for i in df['Label']:
        all_locations.update(i.split(','))
    all_locations = sorted(list(all_locations))

    rest_indices, testIdx = split_dataset_ensure_label(df, "Label", all_locations, test_size=0.1, min_samples=10)
    k_folds_idx = split_dataset_ensure_label(df.iloc[rest_indices], "Label", all_locations, k=5, min_samples=10)

    labels_list = [locs.split(",") for locs in list(df["Label"])]
    tokenizer = SequenceTokenizer(df['Sequence'], labels_list)
    # tokenizer.save_tokenizer('tokenizer.pkl')

    batch_size = 8 # 32
    learningrate = 0.0001 # 0.001
    epochs = 200
    patience = 20
    device = "cuda"
    thres = 0.5
    checkpoint_folder = "./checkpoints"

    kfold_performance = {"Example_Acc_folds": [], "One_Error_folds": [],
                            "Coverage_folds": [], "Rank_Loss_folds": [], "Hamming_Loss_folds": [],
                            "Average_Precision_folds": [], "Avg_F1_folds": [], "Micro_Precision_folds": [],
                            "Micro_Recall_folds": [], "Avg_AUC_folds": [], "EachAUC_folds": []}

    for fold_idx,(trainIdx,validIdx) in enumerate(k_folds_idx):

        trainDF = df.iloc[trainIdx]
        validDF = df.iloc[validIdx]
        testDF = df.iloc[testIdx]
        train_dataset = UnitLncRNADataset(root='./data_prepared', dataset="v_1",view=f'train_{fold_idx}',df_data=trainDF, tokenizer=tokenizer, foldings=foldings, fea_kmer=features_kmer, fea_cksnap=features_cksnap, emb_k=3, emb_dim=512)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=unit_collate_func)
        
        valid_dataset = UnitLncRNADataset(root='./data_prepared', dataset="v_1",view=f'valid_{fold_idx}',df_data=validDF, tokenizer=tokenizer, foldings=foldings, fea_kmer=features_kmer, fea_cksnap=features_cksnap, emb_k=3, emb_dim=512)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=unit_collate_func)
        test_dataset = UnitLncRNADataset(root='./data_prepared', dataset="v_1",view=f'test',df_data=testDF, tokenizer=tokenizer, foldings=foldings, fea_kmer=features_kmer, fea_cksnap=features_cksnap, emb_k=3, emb_dim=512)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=unit_collate_func)
        
        model = LncTracker(tokenizer=tokenizer, n_features=20, hidden_dim=32, embedding_dim=128, n_classes=tokenizer.label_count,
                            n_conv_layers=4, conv_type="GAT",n_trans_layers=8, head_num=8,
                            dropout=0.2, batch_norm=True,
                            batch_size=batch_size)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=learningrate)
        criterion = torch.nn.BCELoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.3)

        train_valid(model, train_loader, valid_loader, epochs, patience, optimizer, scheduler, criterion, thres, checkpoint_folder, device)
        valid_step(model, test_loader, criterion, thres, device)