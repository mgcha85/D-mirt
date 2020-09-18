import sqlite3
import pandas as pd
import numpy as np
import os

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# import plaidml.keras
# plaidml.keras.install_backend()

from keras.models import Model, Sequential, load_model


class predict:
    def __init__(self, root):
        self.root = root
        # self.peaks = ['DNase-seq', 'H3K4me3', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K9me3']
        self.peaks = ['DNase-seq', 'H3K4me3']

    def get_peak(self, cline):
        # 50kbp / 20bp = 2500
        cols = list(map(str, range(2500)))

        dfs = []
        loc = []
        for i, peak in enumerate(self.peaks):
            fpath = os.path.join(self.root, 'upstream_for_prediction_{}_{}.csv'.format(cline, peak))
            # df = pd.read_csv(fpath)
            df = pd.read_csv(fpath, index_col=0)

            loc.append(df[['chromosome', 'start', 'end', 'strand']])

            df = df[cols]
            dfs.append(df)

        df = pd.concat(dfs)
        loc = pd.concat(loc)
        return pd.concat([loc, df], axis=1).reset_index()

    def convert_sequence(self, sequence, strand):
        data = np.zeros((6, len(sequence)))
        hash = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
        pair = {'A': 'T', 'G': 'C', 'C': 'G', 'T': 'A'}
        for i, ch in enumerate(sequence):
            if ch == 'N':
                continue
            if strand == '-':
                ch = pair[ch]
            data[hash[ch], i] = 1
        if strand == '-':
            data = np.fliplr(data)
        return data

    def get_sequence(self):
        fpath = os.path.join(self.root, 'upstream_for_prediction.db')
        con = sqlite3.connect(fpath)
        return pd.read_sql("SELECT * FROM 'upstream_5k'".format(cline), con, index_col='miRNA')

    def set_data(self, cline):
        df_peak = self.get_peak(cline)
        df_seq = self.get_sequence()
        return df_peak, df_seq

    def run(self, bin, network, cline):
        width = 50          # data size
        stepwidth = 100
        step = int(stepwidth / bin)
        bandwidth = 50000   # upstream 50kb
        n = int(bandwidth / bin)

        model = load_model(os.path.join(self.root, 'model/{}bp_2peaks/{}/model_{}.vgg'.format(bin, network, cline)), compile=False)

        df_peak, df_seq = self.set_data(cline)
        pred_tss = []
        cnt = 0
        df_peak_grp = df_peak.groupby('Name')
        n_dk = len(df_peak_grp)
        for idx, df_peak_sub in df_peak_grp:
            cnt += 1
            print('{} / {}'.format(cnt, n_dk))
            start = df_peak_sub['start'].iloc[0]
            end = df_peak_sub['end'].iloc[0]
            chromosome = df_peak_sub['chromosome'].iloc[0]
            strand = df_peak_sub['strand'].iloc[0]
            offset1 = offset2 = 0

            if idx not in df_seq.index:
                continue

            sequence = df_seq.loc[idx, 'sequence']
            if sequence is None:
                continue
            sequence = self.convert_sequence(sequence.upper(), strand)

            peaks, sequences, loc = [], [], []
            for i in range(0, n, step):
                if start < 0:
                    offset1 = (abs(start) + bin - 1) // bin
                if offset1 >= n - step * bin:
                    offset1 += step
                    offset2 += (step * bin)
                    continue

                peak = df_peak_sub.loc[:, str(offset1):str(offset1+width-1)].values.astype(int)
                psum = peak.sum(axis=1)
                zidx = np.where(psum == 0)[0]
                if peak.shape[1] != width or len(zidx) > 1:
                    offset1 += step
                    offset2 += (step * bin)
                    continue

                seq = sequence[:, offset2:offset2+1000]
                if seq.shape[1] != 1000:
                    offset1 += step
                    offset2 += (step * bin)
                    continue

                peaks.append(peak.T)
                sequences.append(seq.T)
                if strand == '+':
                    loc.append([start + offset2, start + offset2 + 1000])
                else:
                    loc.append([end - 1000 - offset2, end - offset2])
                offset1 += step
                offset2 += (step * bin)

            if peaks and sequences:
                peaks = np.transpose(np.dstack(peaks), (2, 0, 1))
                sequences = np.transpose(np.dstack(sequences), (2, 0, 1))
                results = model.predict([peaks, sequences])

                ridx, cidx = np.where(results > 0.09) # one of them should be greater than 1/11
                if len(ridx) > 0:
                    for r, c in zip(ridx, cidx):
                        prob = results[r, c]
                        if c == 0:
                            continue
                        if strand == '-':
                            c = 49 - c
                        tss_start = loc[r][0] + c * bin
                        pred_tss.append([idx, chromosome, tss_start, tss_start+bin, strand, prob])

        df_pred = pd.DataFrame(pred_tss, columns=['miRNA', 'chromosome', 'start', 'end', 'strand', 'probability'])
        df_pred.to_csv(os.path.join(self.root, 'prediction/{}bp_2peaks/{}'.format(bin, network), 'upstream_pred_tss_{}.csv'.format(cline)), index=False)

    def most_likely_tss(self, bin, network, cline):
        fpath = os.path.join(self.root, 'prediction/{}bp_2peaks/{}'.format(bin, network), 'upstream_pred_tss_{}.csv'.format(cline))
        df = pd.read_csv(fpath)

        # thresholding
        df = df[df['probability'] > 0.8]
        df['tss'] = df['chromosome'] + ';' + df['start'].astype(str) + ';' + df['strand']

        dfs = []
        # it is likely to be predicted multiple times if it is real TSS
        for tss, df_tss in df.groupby('tss'):
            if df_tss.shape[0] > 2:
                dfs.append(df_tss)

        df = pd.concat(dfs)
        df = self.remove_gene_tss(df)
        df.to_csv(fpath.replace('.csv', '_2.csv'), index=False)

    def remove_gene_tss(self, df):
        # protein coding gene
        fpath_pcg = os.path.join(self.root, "database", "gencode.v32lift37.annotation_split.db")
        con = sqlite3.connect(fpath_pcg)

        ridx = []
        for idx in df.index:
            chromosome = df.loc[idx, 'chromosome']
            strand = df.loc[idx, 'strand']
            start = df.loc[idx, 'start']
            end = df.loc[idx, 'end']

            df_pcg = pd.read_sql("SELECT gene_name FROM '{}_{}' WHERE start<{end} AND end>{start}"
                                 "".format(chromosome, strand, start=start, end=end), con)
            if not df_pcg.empty:
                ridx.append(idx)

        if ridx:
            df.drop(ridx, inplace=True)
        return df

    def to_sql(self, bin, network, cline):
        fpath = os.path.join(self.root, 'prediction/{}bp_2peaks/{}'.format(bin, network), 'upstream_pred_tss_{}_2.csv'.format(cline))
        df = pd.read_csv(fpath)

        fpath = os.path.join(self.root, 'prediction', 'pred_tss_{}_{}.db'.format(bin, network))
        con_out = sqlite3.connect(fpath)
        if 'tss' not in df:
            tss = df[['start', 'end']].mean(axis=1).astype(int)
            df['tss'] = df['chromosome'] + ';' + tss.astype(str) + ';' + df['strand']

        df = df.drop_duplicates('tss')
        df.to_sql('upstream_pred_{}'.format(cline), con_out, index=None, if_exists='replace')


if __name__ == '__main__':
    root = "D:/Bioinformatics"
    pred = predict(root)
    bin = 20
    network = 'cnn'

    for cline in ['K562', 'GM12878', 'HelaS3', 'HepG2', 'HES', 'A549', 'MCF7']:
        pred.run(bin, network, cline)
        pred.most_likely_tss(bin, network, cline)
        pred.to_sql(bin, network, cline)
