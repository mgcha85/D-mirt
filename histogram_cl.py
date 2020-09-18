import socket
import sqlite3
import os
import pandas as pd
import numpy as np
from XmlHandler import XmlHandler

import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
prg = cl.Program(ctx, """
    int filterInRange(const int bin_start, const int bin_end, const int start, const int end) {
        if((bin_start < end) && (bin_end >= start))
            return 0;
        else
            return 1;
    }

    void histogram(int ref_start, int ref_end, __global int *data_scr, __global const int *addr_table, __global int *out_histogram, int idx, __global int *width)
    {
        int offset = addr_table[idx*2+0];
        int size = addr_table[idx*2+1];
        
        for(int j=0; j<size; j++) {
            int src_start = data_scr[(offset+j)*2+0];
            int src_end = data_scr[(offset+j)*2+1];
            for(int i=0; i<width[2]; i++) {
                int bin_start = ref_start + (i * width[3]);
                int bin_end = bin_start + width[3];
                if(filterInRange(bin_start, bin_end, src_start, src_end) == 0) {
                    out_histogram[idx*width[2]+i] += 1;
                }
            }
        }    
    }

    void first_filter(int ref_start, int ref_end, __global int *data_scr, int table_size, __global int *addr_table, int idx)
    {
        int cnt = 0;

        for (int i=0; i<table_size; i++) {
            int scr_start = data_scr[i*2+0];
            int scr_end = data_scr[i*2+1];

            if(filterInRange(ref_start, ref_end, scr_start, scr_end) == 0) {
                if(addr_table[idx*2+0] < 0)
                    addr_table[idx*2+0] = i;
                addr_table[idx*2+1] = ++cnt;
            }
        }
    }

    __kernel void cuda_histogram(__global int *data_ref, __global int *data_scr, __global int *addr_table, __global int *out_histogram, __global int *width)
    {
        int idx = get_global_id(0);
        if (idx >= width[0]) return;

        int start = data_ref[idx*2+0];
        int end = data_ref[idx*2+1];

        first_filter(start, end, data_scr, width[1], addr_table, idx);
        histogram(start, end, data_scr, addr_table, out_histogram, idx, width);

    }
""").build()


class histogram_gpu:
    def __init__(self, user_param):
        self.user_param = user_param

    def as_batch(self, df, BATCH_SIZE):
        N = df.shape[0]
        M = np.ceil((N + BATCH_SIZE - 1) // BATCH_SIZE)

        for i in range(int(M)):
            sidx = i * BATCH_SIZE
            eidx = sidx + BATCH_SIZE
            if eidx > N:
                eidx = N
            yield df[sidx: eidx]

    def histogram_gpu(self, df_ref, dbpath_src, src_tname):
        histogram_width = int(self.user_param['numeric']['bandwidth'] // self.user_param['numeric']['bin'])

        con = sqlite3.connect(dbpath_src)
        df_ref_chr = df_ref.groupby('chromosome')

        dfs = []
        for i, (chr, df_ref_sub) in enumerate(df_ref_chr):
            if len(chr) > 5:
                continue
            df_ref_sub_str = df_ref_sub.groupby('strand')

            for str, df_ref_sub_sub in df_ref_sub_str:
                print(chr, str)
                N = np.int32(df_ref_sub_sub.shape[0])
                df__ = pd.read_sql("SELECT start, end FROM '{}_{}_{}'".format(src_tname, chr, str), con)
                df_batch = self.as_batch(df__, BATCH_SIZE=1 << 20)
                out_histogram__ = np.zeros((N, histogram_width)).astype(np.int32)

                # inputs to gpu
                for df in df_batch:
                    data_ref = df_ref_sub_sub[['start', 'end']].values.flatten().astype(np.int32)

                    # malloc to gpu
                    data_ref_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_ref)
                    data_scr = df.values.flatten().astype(np.int32)
                    data_scr_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_scr)
                    addr_table = -np.ones((N, 2)).flatten().astype(np.int32)
                    addr_table_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=addr_table)
                    out_histogram = np.zeros((N, histogram_width)).flatten().astype(np.int32)
                    out_histogram_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=out_histogram)

                    M = np.int32(df.shape[0])
                    width = np.array([N, M, histogram_width, self.user_param['numeric']['bin']]).astype(np.int32)
                    width_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=width)

                    prg.cuda_histogram(queue, (N, ), None, data_ref_g, data_scr_g, addr_table_g, out_histogram_g, width_g)
                    cl.enqueue_copy(queue, out_histogram, out_histogram_g)

                    out_histogram = out_histogram.reshape((-1, histogram_width))
                    out_histogram__ += out_histogram

                df_add = pd.DataFrame(data=out_histogram__, columns=range(histogram_width))
                df = pd.concat([df_ref_sub_sub[['start', 'end']].reset_index(drop=True), df_add], axis=1)
                df.loc[:, 'chromosome'] = chr
                df.loc[:, 'strand'] = str
                df.index = df_ref_sub_sub.index
                dfs.append(df[['chromosome', 'start', 'end', 'strand', *df_add.columns]])
        return pd.concat(dfs).sort_index()

    def run(self, df_ref, dbpath_src, src_tname):
        hbw = int(self.user_param['numeric']['bandwidth']) / 2

        df_ref[['start', 'end']] = df_ref[['start', 'end']].astype(int)
        tss = df_ref[['start', 'end']].mean(axis=1).astype(np.int32)
        df_ref.loc[:, 'start'] = tss - hbw
        df_ref.loc[:, 'end'] = tss + hbw

        df_ref[['start', 'end']] = df_ref[['start', 'end']].astype(int)
        df_res = self.histogram_gpu(df_ref, dbpath_src, src_tname)
        return df_res
