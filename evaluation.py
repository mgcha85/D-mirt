import sqlite3
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from XmlHandler import XmlHandler


class evaluation:
    def __init__(self, root):
        self.root = root

    def pro(self, cell_lines=['GM12878', 'K562']):
        from histogram_cl import histogram_gpu
        fpath = os.path.join(self.root, 'database', 'PRO.db')
        con = sqlite3.connect(fpath)

        for cell_line in cell_lines:
            df_ref = pd.read_sql("SELECT * FROM '{}'".format(cell_line), con)
            df_ref['tss'] = df_ref[['tss_start', 'tss_stop']].mean(axis=1).astype(int)
            df_ref['start'] = df_ref['tss'].astype(int) - 500
            df_ref['end'] = df_ref['tss'].astype(int) + 500

            gro_path = os.path.join(self.root, 'database', 'GRO_cap.db')
            hgpu = histogram_gpu(XmlHandler.load_param("user_param.xml"))
            df = hgpu.run(df_ref, gro_path, '_'.join([cell_line, 'hg19']))

            dirname = os.path.join(self.root, 'evaluation')
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            out_con = sqlite3.connect(os.path.join(dirname, 'eval_peaks.db'))
            df.to_sql('PRO_GRO_{}'.format(cell_line), out_con, if_exists='replace', index=None)

    def hua(self, cell_lines=['GM12878', 'K562']):
        from histogram_cl import histogram_gpu
        fpath = os.path.join(self.root, 'database', 'Supplementary file4-alternative_TSS.db')
        con = sqlite3.connect(fpath)

        for cell_line in cell_lines:
            df_ref = pd.read_sql("SELECT * FROM 'cell_specific' WHERE cell_lines LIKE '%{}%'".format(cell_line), con)
            df_ref['start'] = df_ref['tss'].astype(int) - 500
            df_ref['end'] = df_ref['tss'].astype(int) + 500

            gro_path = os.path.join(self.root, 'database', 'GRO_cap.db')
            hgpu = histogram_gpu(XmlHandler.load_param("user_param.xml"))
            df = hgpu.run(df_ref, gro_path, '_'.join([cell_line, 'hg19']))

            dirname = os.path.join(self.root, 'evaluation')
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            out_con = sqlite3.connect(os.path.join(dirname, 'eval_peaks.db'))
            df.to_sql('HUA_GRO_{}'.format(cell_line), out_con, if_exists='replace', index=None)

    def dmirt(self, cell_lines=['GM12878', 'K562']):
        from histogram_cl import histogram_gpu

        fpath = os.path.join(self.root, 'prediction', 'pred_tss_20_cnn.db')
        con = sqlite3.connect(fpath)
        for cell_line in cell_lines:
            print(cell_line)
            df_ref = pd.read_sql("SELECT miRNA, chromosome, start, end, strand FROM "
                                       "(SELECT * FROM 'upstream_pred_{}')".format(cell_line), con)
            df_ref[['start', 'end']] = df_ref[['start', 'end']].astype(int)

            tss = df_ref[['start', 'end']].mean(axis=1).astype(int)
            df_ref['start'] = tss - 500
            df_ref['end'] = tss + 500

            gro_path = os.path.join(self.root, 'database', 'GRO_cap.db')
            hgpu = histogram_gpu(XmlHandler.load_param("user_param.xml"))
            df = hgpu.run(df_ref[['chromosome', 'start', 'end', 'strand']], gro_path, '_'.join([cell_line, 'hg19']))

            df['miRNA'] = df_ref['miRNA']
            nidx = df[df['strand'] == '-'].index
            hist_col = list(range(50))
            contents = df.loc[nidx, hist_col[::-1]]
            contents.columns = hist_col
            df.loc[nidx, hist_col] = contents

            dirname = os.path.join(self.root, 'evaluation')
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            out_con = sqlite3.connect(os.path.join(dirname, 'eval_peaks.db'))
            df.to_sql('DMIRT_GRO_{}'.format(cell_line), out_con, index=None, if_exists='replace')

    def h3k4me3(self, paper='DMIRT', cell_lines=['GM12878', 'K562']):
        from histogram_cl import histogram_gpu

        out_con = sqlite3.connect(os.path.join(self.root, 'evaluation', 'eval_peaks.db'))
        for cell_line in cell_lines:
            fpath = os.path.join(self.root, 'prediction', 'pred_tss_20_cnn.db')
            con = sqlite3.connect(fpath)

            if paper == 'DMIRT':
                df_ref = pd.read_sql_query("SELECT miRNA, chromosome, start, end, strand FROM "
                                           "(SELECT * FROM 'upstream_pred_{}')".format(cell_line), con)
                df_ref[['start', 'end']] = df_ref[['start', 'end']].astype(int)
                tss = df_ref[['start', 'end']].mean(axis=1).astype(int)
                mir_label = 'miRNA'
            elif paper == 'HUA':
                fpath = os.path.join(self.root, 'database', 'Supplementary file4-alternative_TSS.db')
                con = sqlite3.connect(fpath)
                df_ref = pd.read_sql_query("SELECT * FROM 'cell_specific' WHERE cell_lines LIKE '%{}%'".format(cell_line), con)
                tss = df_ref['tss'].astype(int)
                mir_label = '#MIR'
            elif paper == 'PRO':
                fpath = os.path.join(self.root, 'database', 'PRO.db')
                con = sqlite3.connect(fpath)
                df_ref = pd.read_sql_query("SELECT * FROM '{}'".format(cell_line), con)
                tss = df_ref[['tss_start', 'tss_stop']].mean(axis=1).astype(int)
                mir_label = 'miRNA'
            else:
                fpath = os.path.join(self.root, 'prediction', 'pred_tss_20_cnn.db')
                con = sqlite3.connect(fpath)
                df_ref = pd.read_sql("SELECT miRNA, chromosome, start, end, strand FROM "
                                           "(SELECT * FROM 'upstream_pred_{}')".format(cell_line), con)
                df_ref[['start', 'end']] = df_ref[['start', 'end']].astype(int)
                tss = df_ref[['start', 'end']].mean(axis=1).astype(int)
                mir_label = 'miRNA'

            df_ref['start'] = tss - 500
            df_ref['end'] = tss + 500

            fpath = os.path.join(self.root, 'database', 'bioinfo_{}.db'.format(cell_line))
            hgpu = histogram_gpu(XmlHandler.load_param("user_param.xml"))

            df_rsc = pd.read_excel('fids.xlsx', index_col=0)
            fid = df_rsc.loc[cell_line, 'H3K4me3']
            df = hgpu.run(df_ref[['chromosome', 'start', 'end', 'strand']], fpath, fid)
            if mir_label:
                df['miRNA'] = df_ref[mir_label]

            nidx = df[df['strand'] == '-'].index
            hist_col = list(range(50))
            contents = df.loc[nidx, hist_col[::-1]]
            contents.columns = hist_col
            df.loc[nidx, hist_col] = contents

            df.to_sql('{}_h3k4me3_{}'.format(paper, cell_line), out_con, index=None, if_exists='replace')

    def cage_tag(self, paper='DMIRT', cell_lines=['GM12878', 'K562']):
        from histogram_cl import histogram_gpu

        for cell_line in cell_lines:
            if paper == 'DMIRT':
                fpath = os.path.join(self.root, 'prediction', 'pred_tss_20_cnn.db')
                con = sqlite3.connect(fpath)
                df_ref = pd.read_sql("SELECT miRNA, chromosome, start, end, strand FROM "
                                           "(SELECT * FROM 'upstream_pred_{}')".format(cell_line), con)
                df_ref[['start', 'end']] = df_ref[['start', 'end']].astype(int)
                tss = df_ref[['start', 'end']].mean(axis=1).astype(int)
                mir_label = 'miRNA'
            elif paper == 'HUA':
                fpath = os.path.join(self.root, 'database', 'Supplementary file4-alternative_TSS.db')
                con = sqlite3.connect(fpath)
                df_ref = pd.read_sql("SELECT * FROM 'cell_specific' WHERE cell_lines LIKE '%{}%'".format(cell_line), con)
                tss = df_ref['tss'].astype(int)
                mir_label = '#MIR'
            elif paper == 'PRO':
                fpath = os.path.join(self.root, 'database', 'PRO.db')
                con = sqlite3.connect(fpath)
                df_ref = pd.read_sql("SELECT * FROM '{}'".format(cell_line), con)
                tss = df_ref[['tss_start', 'tss_stop']].mean(axis=1).astype(int)
                mir_label = 'miRNA'
            else:
                fpath = os.path.join(self.root, 'prediction', 'pred_tss_20_cnn.db')
                con = sqlite3.connect(fpath)
                df_ref = pd.read_sql("SELECT miRNA, chromosome, start, end, strand FROM "
                                           "(SELECT * FROM 'upstream_pred_{}')".format(cell_line), con)
                df_ref[['start', 'end']] = df_ref[['start', 'end']].astype(int)
                tss = df_ref[['start', 'end']].mean(axis=1).astype(int)
                mir_label = 'miRNA'

            df_ref['start'] = tss - 500
            df_ref['end'] = tss + 500

            tss = df_ref[['start', 'end']].mean(axis=1).astype(int)

            df_ref['start'] = tss - 500
            df_ref['end'] = tss + 500

            hgpu = histogram_gpu(XmlHandler.load_param("user_param.xml"))
            fpath = os.path.join(self.root, "database", "hCAGE_ctss.db")
            rsc_tname = '{}_hg19_ctss'.format(cell_line)
            df = hgpu.run(df_ref[['chromosome', 'start', 'end', 'strand']], fpath, rsc_tname)
            if mir_label:
                df['miRNA'] = df_ref[mir_label]

            nidx = df[df['strand'] == '-'].index
            hist_col = list(range(50))
            contents = df.loc[nidx, hist_col[::-1]]
            contents.columns = hist_col
            df.loc[nidx, hist_col] = contents

            out_con = sqlite3.connect(os.path.join(self.root, 'evaluation', 'eval_peaks.db'))
            df.to_sql('{}_CAGE_{}'.format(paper, cell_line), out_con, index=None, if_exists='replace')


if __name__ == '__main__':
    root = "D:/DmiRT"
    eg = evaluation(root)

    cell_lines = ['GM12878', 'K562']

    # set GRO data to compare prediction result
    eg.pro(cell_lines=cell_lines)
    eg.hua(cell_lines=cell_lines)
    eg.dmirt(cell_lines=cell_lines)

    for p in ['PRO', 'DMIRT', 'HUA']:
        eg.h3k4me3(paper=p, cell_lines=cell_lines)
        eg.cage_tag(paper=p, cell_lines=cell_lines)
