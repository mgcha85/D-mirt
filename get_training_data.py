import sqlite3
import pandas as pd
import os
from Database import Database
import pickle as pkl
import socket
import numpy as np
from numpy import matlib
from copy import deepcopy
import random


class train:
    def __init__(self, root):
        self.root = root

    def merge_all_data(self, bin=20):
        from joblib import Parallel, delayed
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        if num_cores > 16:
            num_cores = 16

        def processInput(i):
            print('batch {}'.format(i))
            df_peak_sub_spt = df_peak_sub.iloc[i * batch_size: (i + 1) * batch_size]
            df_peak_sub_spt.set_index('index')['tss_bin'].to_csv(lab_path_out.format(i, cline))
            df_peak_sub_spt.set_index('index').drop(['chromosome', 'start', 'end', 'strand', 'cell_line', 'tss_bin'],
                                                    axis=1).to_csv(peak_path_out.format(i, cline))

            seq = []
            indecies = set()
            for idx in df_peak_sub_spt.index:
                index = df_peak_sub_spt.loc[idx, 'index']
                if index in indecies:
                    continue

                chr, tss, strand = index.split(';')
                tag = '_'.join([chr, strand])
                seq.append(seq_data[tag].loc[index:index])
                indecies.add(index)

            df_seq = pd.concat(seq)
            df_seq.to_csv(seq_path_out.format(i, cline))

        con = sqlite3.connect(os.path.join(self.root, 'histone_features_{}bp.db'.format(bin)))
        # 7 epigenomic
        # activation = ['DNase-seq', 'H3K4me3', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K9me3']
        # 2 epigenomic
        activation = ['DNase-seq', 'H3K4me3']
        n = len(activation)

        lab_path_out = os.path.join(self.root, 'P/{}bp_{}peaks'.format(bin, n), 'trn_data_{}_{}_lab.csv')
        peak_path_out = os.path.join(self.root, 'P/{}bp_{}peaks'.format(bin, n), 'trn_data_{}_{}_peak.csv')
        seq_path_out = os.path.join(self.root, 'P/{}bp_{}peaks'.format(bin, n), 'trn_data_{}_{}_seq.csv')

        tnames = []
        for tname in Database.load_tableList(con):
            for act in activation:
                if act in tname:
                    tnames.append(tname)

        dfs = []
        clines = set()
        for tname in tnames:
            cline, act, _ = tname.split('_')
            clines.add(cline)

            print('load peak {}, {}'.format(cline, act))
            df = pd.read_sql("SELECT * FROM '{}'".format(tname), con)
            tss = df[['start', 'end']].mean(axis=1).astype(int)

            df['cell_line'] = cline
            df['label'] = act
            df['index'] = df['chromosome'] + ';' + tss.astype('str') + ';' + df['strand']

            df = df.drop(['gene_name', 'transcript_name'], axis=1)
            dfs.append(df)

        # merge all data
        df_peak = pd.concat(dfs).sort_values(by=['index', 'label'])
        del dfs

        # shuffle
        dfs = []
        for _, df_grp in df_peak.groupby('index'):
            dfs.append(df_grp)

        random.shuffle(dfs)
        df_peak = pd.concat(dfs).reset_index(drop=True)
        del dfs

        seq_data = {}
        for chr in set(df_peak['chromosome']):
            for str in set(df_peak['strand']):
                print('load sequence {}, {}'.format(chr, str))
                tag = '_'.join([chr, str])
                fpath = os.path.join(self.root, 'training_sequence_{}.csv'.format(tag))
                seq_data[tag] = pd.read_csv(fpath, header=None)
                loc = seq_data[tag][0].str.split(';', expand=True)
                tss = loc[[1, 2]].astype(int).mean(axis=1).astype(int)
                seq_data[tag][0] = loc[0] + ';' + tss.astype('str') + ';' + loc[3]
                seq_data[tag] = seq_data[tag].set_index(0, drop=True)

        n_split = 100
        for cline in clines:
            print(cline)
            df_peak_sub = df_peak[df_peak['cell_line'] == cline]
            batch_size = int((df_peak_sub.shape[0] + n_split - 1) / n_split)
            # for i in range(n_split):
            #     processInput(i)
            Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in range(n_split))

    def get_seq_train(self, chr, str):
        dfs = {}
        for type in ['conservation', 'sequence']:
            fpath = os.path.join(self.root, type, '{}_{}_{}.csv'.format(type, chr, str))
            dfs[type] = pd.read_csv(fpath, index_col=0)

        data = np.zeros((5*dfs['conservation'].shape[0], 1000))
        index = []
        for i, idx in enumerate(dfs['conservation'].index):
            data[i*5:4+i*5, :] = dfs['sequence'].loc[idx]
            data[4+i*5, :] = dfs['conservation'].loc[idx]
            index += [idx] * 5
        df_res = pd.DataFrame(data=data, index=index)
        df_res.to_csv(os.path.join(self.root, 'training_sequence_{}_{}.csv'.format(chr, str)), header=False)

    def get_histone_train(self, cell_lines, histones, bin):
        fname = os.path.join(self.root, 'histone_features_{}bp.db'.format(bin))
        con = sqlite3.connect(fname)
        con_out = sqlite3.connect(os.path.join(self.root, 'training_{}bp.db'.format(bin)))

        cols = list(map(str, range(50)))
        n = len(histones)

        for cell_line in cell_lines:
            print(cell_line)

            dfs, res_idx = [], []
            for hi in histones:
                sql = "SELECT * FROM '{}_{}'".format(cell_line, hi)
                df = pd.read_sql(sql, con)
                df['tss'] = df[['start', 'end']].mean(axis=1).astype('int')
                df.index = df['chromosome'] + ';' + df['tss'].astype(str) + ';' + df['strand']

                df_str = []
                for strand, df_sub in df.groupby('strand'):
                    if strand == '-':
                        df_sub[cols] = np.fliplr(df_sub[cols].values)
                    df_str.append(df_sub)
                df = pd.concat(df_str)

                df = df[cols]
                df_bin = deepcopy(df)
                df_bin[df_bin != 0] = 1

                dfs.append(df)
                dfs.append(df_bin)
                res_idx.append(hi)
                res_idx.append(hi + '_bin')

            dfs = pd.concat(dfs, axis=1)
            length = dfs.shape[0]
            data = dfs.values.reshape(2 * n * length, -1)

            index = matlib.repmat(dfs.index.values, 2 * n, 1).T.flatten()
            df_res = pd.DataFrame(data, index=index)
            df_res.index.name = 'loc'

            label = matlib.repmat(np.array(res_idx), 1, length).flatten()
            df_res['label'] = label
            df_res['cell_line'] = cell_line
            df_res.to_sql('peak', con_out, if_exists='append')

    def get_label(self, cell_lines, histones, bin):
        fname = os.path.join(self.root, 'histone_features_{}bp.db'.format(bin))
        con = sqlite3.connect(fname)
        con_out = sqlite3.connect(os.path.join(self.root, 'training_{}bp.db'.format(bin)))

        for cell_line in cell_lines:
            print(cell_line)

            for hi in histones:
                df = pd.read_sql("SELECT chromosome, start, end, strand, tss_bin FROM '{}_{}'".format(cell_line, hi), con)
                tss_bin = df['tss_bin'].astype(int)

                data = np.zeros((df.shape[0], 50))
                data[tss_bin.index, tss_bin] = 1

                df['tss'] = df[['start', 'end']].mean(axis=1).astype('int')
                df.index = df['chromosome'] + ';' + df['tss'].astype(str) + ';' + df['strand']
                df_res = pd.DataFrame(data=data, index=df.index)
                df_res['cell_line'] = cell_line
                df.index.name = 'loc'
                df_res.to_sql('label', con_out, if_exists='append')


if __name__ == '__main__':
    root = "D:/Bioinformatics"
    tr = train(root)
    cell_lines = ['A549', 'GM12878', 'HES', 'HelaS3', 'HepG2', 'K562', 'MCF7']
    histones = ['DNase-seq', 'H3K4me3', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K9me3']

    for bin in [20, 100]:
        tr.get_histone_train(cell_lines, histones, bin)
        tr.get_label(cell_lines, histones, bin)
        tr.merge_all_data(bin)
