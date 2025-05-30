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
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default="test", help='Path to the input dataset used for training the model.')
parser.add_argument('--output_path', default="test", help='Path to save the prediction results or output files.')
parser.add_argument('--batch_size', type=int, default=32, help='Number of samples per training batch.')
parser.add_argument('--isMultiLabel', type=bool, default=True, help='Whether the task is a multi-label classification problem.')
parser.add_argument('--isAutoThres', type=bool, default=True, help='Whether to automatically determine the threshold for prediction.')
parser.add_argument('--device', default="cpu", choices=["cpu", "cuda"], help='Device to use for computation: "cpu" or "cuda".')
args = parser.parse_args()


input_path = args.input_path
output_path = args.output_path
batch_size = args.batch_size
isMultiLabel = args.isMultiLabel
isAutoThres = args.isAutoThres
device = args.device

checkpoint_folder = "./checkpoints"

# device = "cuda"
if __name__ == '__main__':
    folding_path = './features/lnctracker_foldings.pkl'
    cksnap_path = './features/lnctracker_cksnap.pkl'
    kmer_path = './features/lnctracker_5mer.pkl'

    test_df = utils.read_fasta_to_df(input_path)

    # If the features of the sequence have not been computed yet, cancel the following comment markers

    # out_fasta_dot = "temp_fasata_dot"
    # utils.linear_fold(list(testDF['Sequence']), list(testDF['Description']), out_fasta_dot)
    # utils.dot_fasta_to_pkl(out_fasta_dot+".fasta", folding_path)

    # kw = {'order': 'ACGT'}
    # cksnap_encoding = utils.CKSNAP(input_path, gap=5, **kw)
    # kmer_encoding = utils.Kmer(input_path, k=5, type="RNA", upto=False, normalize=True, **kw)
    # with open(cksnap_path, 'wb') as handle:
    #     pickle.dump(cksnap_encoding, handle)
    # with open(kmer_path, 'wb') as handle:
    #     pickle.dump(kmer_encoding, handle)

    foldings = pickle.load(open(folding_path, "rb"))
    features_cksnap = pickle.load(open(cksnap_path, "rb"))
    features_kmer = pickle.load(open(kmer_path, "rb"))


    tokenizer = load_tokenizer(os.path.join(checkpoint_folder, "tokenizer.pkl") )

    if isMultiLabel:
        if isAutoThres:

            ref_df = utils.read_fasta_to_df("./data/lncRNA.fasta")

            labels_list = [locs.split(",") for locs in list(ref_df["Label"])]

            all_labels = [label.strip() for sublist in labels_list for label in sublist]

            label_counter = Counter(all_labels)
            total_samples = len(ref_df)

            mean_freq = np.mean(list(label_counter.values())) / total_samples

            thresholds = {
                label: float(np.clip(0.5 +  0.7 * ((count / total_samples) - mean_freq), 0.1, 0.9))
                for label, count in label_counter.items()
            }
            
            thres = np.array([thresholds[lab] for lab in list(tokenizer.mlb.classes_)])
            print(f"Thresholds: {thresholds}")
            
        else:
            thres = 0.5
    else:
        thres = 0.5

    test_dataset = UnitLncRNADataset(root='./data_prepared', dataset="test",view=f'v1',df_data=test_df, tokenizer=tokenizer, foldings=foldings, fea_kmer=features_kmer, fea_cksnap=features_cksnap, emb_k=3, emb_dim=512, isMultiLabel=isMultiLabel, device=device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=unit_collate_func)

    model = LncTracker(tokenizer=tokenizer, n_features=20, hidden_dim=128, embedding_dim=128, n_classes=tokenizer.label_count,
                            n_conv_layers=4, conv_type="GAT",n_trans_layers=8, head_num=8,
                            dropout=0.2, batch_norm=True,
                            batch_size=batch_size, activaton_function="sigmoid")
    
    model.load_state_dict(torch.load(os.path.join(checkpoint_folder, "model_final.pth"), map_location=device))
    model.to(device)
    res_binary, res_prob = predict(model, test_loader, thres=thres, device=device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = ["Seq_ID"] + list(tokenizer.mlb.classes_)
        writer.writerow(["Type"] + header)

        # 写入概率预测
        for row in res_prob:
            writer.writerow(["Prob"] + row.tolist())

        # 写入二值预测
        for row in res_binary:
            writer.writerow(["Binary"] + row.tolist())
