import os, re, sys
import pickle
import pandas as pd
import numpy as np
from Bio import SeqIO
import itertools
from collections import Counter
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,StratifiedKFold,KFold

def read_fasta_to_df(file_path):
    records = list(SeqIO.parse(file_path, 'fasta'))
    df = pd.DataFrame({"Description":[str(record.description).split("|")[0] for record in records], 'Sequence': [str(record.seq) for record in records], 'Label': [",".join(str(record.id).split('|')[-1].split(',')) for record in records]})
    return df

def linear_fold(sequences, ids, out_fasta_name):
    for seq, id in zip(sequences, ids):
        # if not id in processed_ids:
        print(f"{id} is processing...")
        with open("tmp.fasta", "w") as ofile: 
            ofile.write(f">{id}\n{seq}\n")
        os.system(f"cat tmp.fasta | ./linearfold_v > tmp.dot") 
        in_lines = open("tmp.dot","r").readlines()
        with open("clean_tmp.dot","w") as out_file:
            for line in in_lines:
                if ">" in line: # extract just the ID
                    out_file.write(':'.join(line.split(":")[1:]).strip() + "\n")
                    # out_file.write(line+ "\n")
                else:
                    out_file.write(line)
        os.system("cat " + "clean_tmp.dot" + " >> " + out_fasta_name + ".fasta") 

def dot_fasta_to_pkl(file, out_pkl):
    with open(file) as f:
        records=f.read()
    if re.search('>', records) == None:
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]
    # print(records)
    seq_dotbracket = {} # 
    for fasta in records:
        valueList=[]
        array = fasta.split('\n')
        sequence,dot_bracket =array[1],array[2]
        sequence = re.sub('U', 'T', sequence)
        if 'N' in sequence:
            sequence = re.sub('N', 'G', sequence)
            print(array[0])
        dot_bracket_list=dot_bracket.split()
        # print(dot_bracket_list)
        # break
        ev=float(dot_bracket_list[1].split('(')[1].split(')')[0]) #.replace('\U00002013', '-')
        valueList.append(dot_bracket_list[0])
        valueList.append(ev)
        seq_dotbracket[sequence]=valueList
    with open(out_pkl, 'wb') as handle:
        pickle.dump(seq_dotbracket, handle)

def get_min_sequence_length(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(i[1]):
            minLen = len(i[1])
    return minLen

def read_nucleotide_sequences(file):
    if os.path.exists(file) == False:
        print('Error: file %s does not exist.' % file)
        sys.exit(1)
    with open(file) as f:
        records = f.read()
    if re.search('>', records) == None:
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]
    fasta_sequences = []
    for fasta in records:
        array = fasta.split('\n')
        header, sequence = array[0].split()[0], re.sub('[^ACGTU]', '-', ''.join(array[1:]).upper())
        header_array = header.split('|')
        name = header_array[0]
        label = header_array[-1] if len(header_array) >= 2 else '0'
        label_train = header_array[-1] if len(header_array) >= 3 else 'training'
        sequence = re.sub('U', 'T', sequence)
        fasta_sequences.append([name, sequence, label, label_train])
    return fasta_sequences

def CKSNAP(fastas, gap=5, **kw):
    fastas = read_nucleotide_sequences(fastas)
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if get_min_sequence_length(fastas) < gap + 2:
        print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap + 2) + '\n\n')
        return 0

    AA = kw['order'] if kw['order'] != None else 'ACGT'
    # encodings = []
    encodings = {}
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)

    for i in fastas:
        seq_key=i[1]
        name, sequence, label = i[0], i[1], i[2]
        # code = [name, label]
        code=[]
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                    index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        # encodings.append(code)
        encodings[seq_key]=code
    return encodings

def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        kmer.append(sequence[i:i + k])
    return kmer

def Kmer(fastas, k=2, type="DNA", upto=False, normalize=True, **kw):
    # encoding = []
    fastas = read_nucleotide_sequences(fastas)
    encoding = {}
    header = ['#', 'label']
    NA = 'ACGT'
    if type in ("DNA", 'RNA'):
        NA = 'ACGT'
    else:
        NA = 'ACDEFGHIKLMNPQRSTVWY'
    if k < 1:
        print('Error: the k-mer value should larger than 0.')
        return 0
    if upto == True:
        for tmpK in range(1, k + 1):
            for kmer in itertools.product(NA, repeat=tmpK):
                header.append(''.join(kmer))
        # encoding.append(header)
        for i in fastas:
            seq_key=i[1]
            name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
            count = Counter()
            for tmpK in range(1, k + 1):
                kmers = kmerArray(sequence, tmpK)
                count.update(kmers)
                if normalize == True:
                    for key in count:
                        if len(key) == tmpK:
                            count[key] = count[key] / len(kmers)
            # code = [name, label]
            code=[]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding[seq_key]=code
            # encoding.append(code)
    else:
        for kmer in itertools.product(NA, repeat=k):
            header.append(''.join(kmer))
        # encoding.append(header)
        for i in fastas:
            seq_key=i[1]
            name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
            kmers = kmerArray(sequence, k)
            count = Counter()
            count.update(kmers)
            if normalize == True:
                for key in count:
                    count[key] = count[key] / len(kmers)
            # code = [name, label]
            code = []
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            # encoding.append(code)
            encoding[seq_key]=code
    return encoding

def split_dataset_ensure_label(df, labels_column, locations, k=None, test_size=0.2, min_samples=10):

    df = df.copy() 
    X = df['Sequence'].values

    def encode_labels(label_string):
        labels = label_string.split(',')
        return [1 if label in labels else 0 for label in locations]

    y = np.array([encode_labels(label) for label in df[labels_column]])
    n_labels = y.shape[1]

    y_indices = {i: np.where(y[:, i] == 1)[0] for i in range(n_labels)}

    def _ensure_min_samples(indices):
        indices = set(indices)
        for i in range(n_labels):
            label_samples = np.intersect1d(list(indices), y_indices[i])
            missing_count = min_samples - len(label_samples)

            if missing_count > 0:
                additional_samples = set(y_indices[i]) - indices
                indices.update(list(additional_samples)[:missing_count])

        return np.array(list(indices))

    if k is not None:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        folds = []

        for train_index, test_index in kf.split(X):
            test_index = _ensure_min_samples(test_index) 
            train_index = np.setdiff1d(train_index, test_index) 

            folds.append((train_index, test_index))

        return folds

    else:
        test_indices = set()
        df['label_list'] = df[labels_column].str.split(',')

        for label in locations:
            label_samples = df[df['label_list'].apply(lambda x: label in x)]
            selected_samples = (
                label_samples.index if len(label_samples) <= min_samples 
                else label_samples.sample(n=min_samples, random_state=42).index
            )
            test_indices.update(selected_samples)

        remaining_samples = df.drop(index=test_indices)
        extra_train, extra_test = train_test_split(
            remaining_samples, test_size=test_size, random_state=42
        )

        test_indices.update(extra_test.index)
        train_indices = np.setdiff1d(df.index, list(test_indices)) 

        return np.array(train_indices), np.array(list(test_indices))
    
