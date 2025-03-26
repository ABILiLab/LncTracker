# LncTracker: a unified multi-channel framework for multi-label lncRNA localization
## Introduction

Long non-coding RNAs (lncRNAs) play essential roles in various biological processes, including chromatin modification, cell cycle regulation, transcription, and translation. Recent studies have revealed that the biological functions of lncRNAs are closely associated with their subcellular localizations, making accurate localization prediction critical for understanding their biological roles in cellular regulation and disease mechanisms. However, most existing methods mainly rely on sequence features while neglecting structural information, and they often limited to single-label predictions covering only a small number of subcellular compartments. In this study, we proposed an efficient deep learning framework, LncTracker, for multi-label prediction of lncRNA subcellular localizations across seven distinct compartments. LncTracker adopts a multi-channel architecture that integrates diverse input features into model training, including both primary sequence and secondary structure information. Secondary structures are converted into attributed graphs to capture spatial relationships among nucleotides, including adjacency and base-pairing connections. These structural features are then combined with sequence-based features to predict subcellular localization probabilities. Such a design enables LncTracker to learn joint representations of sequences and structures, thereby enhancing predictive performance and robustness. Benchmarking experiments demonstrated the superiority of LncTracker over state-of-the-art approaches, particularly in handling imbalanced localization scenarios. 

## Environment
* Anaconda
* python 3.8.18

## Dependency

* numpy 1.24.4
* pandas 1.5.3
* scikit-learn 1.2.2
* scipy 1.10.1
* torch 2.0.1
* torch-geometric 2.4.0
* biopython 1.83
* LinearFold

## Prediction

python predict.py --input_path ./data/example.fasta --output_path results.csv --device cpu

