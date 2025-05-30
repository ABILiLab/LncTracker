import argparse
import pandas as pd
import numpy as np
import pickle
import os
import sys
from model import *
from data import SequenceTokenizer, UnitLncRNADataset, unit_collate_func
from torch.utils.data import DataLoader
from metrics import *
import utils
from utils import *
from data import *
from Bio import SeqIO
import torch
import torch.optim as optim
from datetime import datetime

parser = argparse.ArgumentParser(description="Train a model with specified configurations.")

parser.add_argument('--input_path', default="test", help='Path to the input dataset used for training the model.')
parser.add_argument('--batch_size', type=int, default=32, help='Number of samples per training batch.')
parser.add_argument('--learningrate', type=float, default=0.0001, help='Learning rate for the optimizer.')
parser.add_argument('--epochs', type=int, default=200, help='Maximum number of training epochs.')
parser.add_argument('--patience', type=int, default=20, help='Number of epochs to wait for improvement before early stopping.')
parser.add_argument('--isMultiLabel', type=bool, default=True, help='Whether the task is a multi-label classification problem.')
parser.add_argument('--isAutoThres', type=bool, default=False, help='Whether to automatically determine the threshold for prediction.')
parser.add_argument('--device', default="cpu", choices=["cpu", "cuda"], help='Device to use for computation: "cpu" or "cuda".')

args = parser.parse_args()

input_path = args.input_path
batch_size = args.batch_size
learningrate = args.learningrate
epochs = args.epochs
patience = args.patience
isMultiLabel = args.isMultiLabel
isAutoThres = args.isAutoThres
device = args.device

checkpoint_folder = "./checkpoints"


if __name__ == '__main__':

    input_path = './data/lncRNA_6000.fasta'

    folding_path = './features/lnctracker_foldings.pkl'
    cksnap_path = './features/lnctracker_cksnap.pkl'
    kmer_path = './features/lnctracker_5mer.pkl'

    df = utils.read_fasta_to_df(input_path)

    # If the features of the sequence have not been computed yet, cancel the following comment markers

    # out_fasta_dot = "./features/temp_fasata_dot"
    # utils.linear_fold(list(df['Sequence']), list(df['Description']), out_fasta_dot)
    # utils.dot_fasta_to_pkl(out_fasta_dot+".fasta", folding_path)
    # print("Folding done.")
    # kw = {'order': 'ACGT'}
    # cksnap_encoding = utils.CKSNAP(input_path, gap=5, **kw)
    # kmer_encoding = utils.Kmer(input_path, k=5, type="RNA", upto=False, normalize=True, **kw)
    # with open(cksnap_path, 'wb') as handle:
    #     pickle.dump(cksnap_encoding, handle)
    # with open(kmer_path, 'wb') as handle:
    #     pickle.dump(kmer_encoding, handle)
    # print("CKSNAP and Kmer encoding done.")

    foldings = pickle.load(open(folding_path, "rb"))
    features_cksnap = pickle.load(open(cksnap_path, "rb"))
    features_kmer = pickle.load(open(kmer_path, "rb"))

    all_locations = set()
    for i in df['Label']:
        all_locations.update(i.split(','))
    all_locations = sorted(list(all_locations))
    print("All locations: ", all_locations)

    rest_indices, testIdx = split_dataset_ensure_label(df, "Label", all_locations, test_size=0.1, min_samples=10)
    k_folds_idx = split_dataset_ensure_label(df.iloc[rest_indices], "Label", all_locations, k=5, min_samples=10)

    # k_folds_idx = split_dataset_ensure_label(df, "Label", all_locations, k=5, min_samples=10)

    labels_list = [locs.split(",") for locs in list(df["Label"])]
    tokenizer = SequenceTokenizer(df['Sequence'], labels_list, isMultiLabel=isMultiLabel)
    tokenizer.save_tokenizer('checkpoints/tokenizer.pkl')
    tokenizer = load_tokenizer('checkpoints/tokenizer.pkl')
    # print("Tokenizer label: ", tokenizer.id2label)

    
    if isMultiLabel:
        if isAutoThres:

            ref_df = utils.read_fasta_to_df("./data/lncRNA.fasta")

            labels_list = [locs.split(",") for locs in list(ref_df["Label"])]

            all_labels = [label.strip() for sublist in labels_list for label in sublist]

            label_counter = Counter(all_labels)
            total_samples = len(ref_df)

            mean_freq = np.mean(list(label_counter.values())) / total_samples
            thresholds = {
                label: float(np.clip(0.5 +  0.8 * ((count / total_samples) - mean_freq), 0.1, 0.9))
                for label, count in label_counter.items()
            }
            
            thres = np.array([thresholds[lab] for lab in list(tokenizer.mlb.classes_)])
            
            thres = 0.5
    else:
        thres = 0.5


    kfold_performance = {"Example_Acc_folds": [], "One_Error_folds": [],
                            "Coverage_folds": [], "Rank_Loss_folds": [], "Hamming_Loss_folds": [],
                            "Average_Precision_folds": [], "Avg_F1_folds": [], "Micro_Precision_folds": [],
                            "Micro_Recall_folds": [], "Avg_AUC_folds": [], "EachAUC_folds": []}


    # all_dataset = UnitLncRNADataset(root='./data_prepared', dataset="v_lnctracker",view=f'all',df_data=df, tokenizer=tokenizer, foldings=foldings, fea_kmer=features_kmer, fea_cksnap=features_cksnap, emb_k=3, emb_dim=512, isMultiLabel=isMultiLabel, device=device)
    # all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=unit_collate_func)
    for fold_idx, (train_idx, valid_idx) in enumerate(k_folds_idx):
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]
        train_dataset = UnitLncRNADataset(root='./data_prepared', dataset=f"{fold_idx}_2",view=f'train',df_data=train_df, tokenizer=tokenizer, foldings=foldings, fea_kmer=features_kmer, fea_cksnap=features_cksnap, emb_k=3, emb_dim=512, isMultiLabel=isMultiLabel, device=device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=unit_collate_func)
        
        valid_dataset = UnitLncRNADataset(root='./data_prepared', dataset=f"{fold_idx}_2",view=f'valid',df_data=valid_df, tokenizer=tokenizer, foldings=foldings, fea_kmer=features_kmer, fea_cksnap=features_cksnap, emb_k=3, emb_dim=512, isMultiLabel=isMultiLabel, device=device)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=unit_collate_func)
        # test_dataset = UnitLncRNADataset(root='./data_prepared', dataset="v_lnclocformer",view=f'test',df_data=testDF, tokenizer=tokenizer, foldings=foldings, fea_kmer=features_kmer, fea_cksnap=features_cksnap, emb_k=3, emb_dim=512)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=unit_collate_func)
        
        model = LncTracker(tokenizer=tokenizer, n_features=20, hidden_dim=128, embedding_dim=128, n_classes=tokenizer.label_count,
                            n_conv_layers=4, conv_type="GAT",n_trans_layers=8, head_num=8,
                            dropout=0.2, batch_norm=True,
                            batch_size=batch_size, activaton_function="sigmoid")
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=learningrate)
        criterion = torch.nn.BCELoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.3)

        best_model = train_valid(model, train_loader, valid_loader, epochs, patience, optimizer, scheduler, criterion, thres, checkpoint_folder, isMultiLabel=isMultiLabel, device=device)
        valid_step(best_model, valid_loader, criterion, thres, isMultiLabel, device)
