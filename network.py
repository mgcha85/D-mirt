import sqlite3
import pandas as pd
import logging
import numpy as np
import pickle as pkl
import socket
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import plaidml.keras
plaidml.keras.install_backend()

from keras.layers import Input, Dense, concatenate, BatchNormalization, Flatten, Conv1D, \
    MaxPooling1D, MaxPooling2D, Dropout, LSTM, Bidirectional
from keras.optimizers import SGD
from keras.models import Model, load_model
from keras.utils import Sequence
from sklearn import metrics


class Generator(Sequence):
    def __init__(self, flist, batch_size, num_peaks, loc=False, type='dual'):
        self.flist = flist
        self.batch_size = batch_size
        self.type = type
        self.loc = loc
        self.num_peaks = num_peaks

    def __len__(self):
        return int((len(self.flist) + self.batch_size - 1) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.flist[idx * self.batch_size:(idx + 1) * self.batch_size]
        peak, sequence, label, loc = [], [], [], []
        for fpath__ in batch_x:
            fpath = fpath__ + '_peak.csv'
            df_peak = pd.read_csv(fpath)

            fpath = fpath__ + '_seq.csv'
            df_seq = pd.read_csv(fpath, index_col=0)

            fpath = fpath__ + '_lab.csv'
            df_lab = pd.read_csv(fpath, index_col=0)

            for idx, df_grp in df_peak.groupby('index'):
                df_grp = df_grp.sort_values('label')

                # if data size is wrong
                if df_grp.shape[0] != self.num_peaks:
                    continue

                # if data size is wrong
                seq = df_seq.loc[idx, :].values
                if seq.shape[0] != 5:
                    continue

                p = df_grp[map(str, range(50))].values
                peak.append(p)

                lab = np.zeros(50)
                lab[df_lab.loc[df_grp['index']].iloc[0]] = 1
                label.append(lab)

                sequence.append(seq)
                loc.append(idx)

        peak, sequence, label, loc = np.dstack(peak).transpose((2, 1, 0)), np.dstack(sequence).transpose((2, 1, 0)), np.vstack(label), np.array(loc)
        if self.loc:
            label = [label.astype(np.int32), loc]
        else:
            label = label.astype(np.int32)

        if self.type == 'dual':
            return [peak.astype(np.float32), sequence.astype(np.float32)], label
        elif self.type == 'peak':
            return [peak.astype(np.float32), label]
        elif self.type == 'sequence':
            return [sequence.astype(np.float32), label]


class Network:
    def __init__(self, root):
        self.root = root

    def lstm_model(self, peak_in1, peak_in2, seq_in):
        peak1 = peak_in1
        peak2 = peak_in2
        seq = seq_in

        peak1 = LSTM(8)(peak1)
        peak2 = LSTM(8)(peak2)
        # peak1 = Bidirectional(LSTM(8))(peak1)
        # peak2 = Bidirectional(LSTM(8))(peak2)

        for fsize in [16, 32, 64, 128]:
            seq = Conv1D(filters=fsize, kernel_size=3, padding="same", activation="relu")(seq)
            seq = Conv1D(filters=fsize, kernel_size=3, padding="same", activation="relu")(seq)
            seq = BatchNormalization()(seq)
            seq = MaxPooling1D(pool_size=2, strides=2)(seq)
        # seq = Bidirectional(LSTM(64))(seq)
        seq = Flatten()(seq)

        merged = concatenate([peak1, peak2, seq])
        s = 4096
        for _ in range(2):
            merged = Dense(s, activation='relu')(merged)
            merged = Dropout(rate=0.5)(merged)
        merged = Dense(50, activation='softmax')(merged)

        model = Model(inputs=[peak_in1, peak_in2, seq_in], outputs=merged)
        model.summary()
        return model

    def sequential_model(self, peak_in, seq_in):
        peak = peak_in
        seq = seq_in
        for fsize in [16, 32, 64, 128]:
            peak = Conv1D(filters=fsize, kernel_size=3, padding="same", activation="relu")(peak)
            peak = Conv1D(filters=fsize, kernel_size=3, padding="same", activation="relu")(peak)
            peak = BatchNormalization()(peak)
            peak = MaxPooling1D(pool_size=2, strides=2)(peak)

        peak = Flatten()(peak)
        peak = Dense(4096, activation='relu')(peak)
        peak = Dropout(rate=0.5)(peak)
        peak = Dense(50, activation='softmax')(peak)
        pmodel = Model(inputs=peak_in, outputs=peak)
        pmodel.summary()

        # seq = Conv2D(64, (2, 5), activation='relu')(seq)
        # seq = BatchNormalization()(seq)
        # seq = MaxPooling2D(pool_size=(1, 4))(seq)

        for fsize in [16, 32, 64, 128]:
            seq = Conv1D(filters=fsize, kernel_size=3, padding="same", activation="relu")(seq)
            seq = Conv1D(filters=fsize, kernel_size=3, padding="same", activation="relu")(seq)
            seq = BatchNormalization()(seq)
            seq = MaxPooling1D(pool_size=2, strides=2)(seq)

        seq = Flatten()(seq)
        seq = Dense(4096, activation='relu')(seq)
        seq = Dropout(rate=0.5)(seq)
        seq = Dense(50, activation='softmax')(seq)
        smodel = Model(inputs=seq_in, outputs=seq)
        smodel.summary()
        return pmodel, smodel

    def model(self, peak_in, seq_in, net_model):
        peak = peak_in
        seq = seq_in
        for fsize in [16, 32, 64, 128]:
            peak = Conv1D(filters=fsize, kernel_size=3, padding="same", activation="relu")(peak)
            peak = Conv1D(filters=fsize, kernel_size=3, padding="same", activation="relu")(peak)
            peak = BatchNormalization()(peak)
            peak = MaxPooling1D(pool_size=2, strides=2)(peak)

            seq = Conv1D(filters=fsize, kernel_size=3, padding="same", activation="relu")(seq)
            seq = Conv1D(filters=fsize, kernel_size=3, padding="same", activation="relu")(seq)
            seq = BatchNormalization()(seq)
            seq = MaxPooling1D(pool_size=2, strides=2)(seq)

        if 'lstm' in net_model:
            peak = Bidirectional(LSTM(32))(peak)
            seq = Bidirectional(LSTM(128))(seq)
        else:
            peak = Flatten()(peak)
            seq = Flatten()(seq)

        merged = concatenate([peak, seq])
        s = 4096
        for _ in range(2):
            merged = Dense(s, activation='relu')(merged)
            merged = Dropout(rate=0.5)(merged)
        merged = Dense(50, activation='softmax')(merged)

        model = Model(inputs=[peak_in, seq_in], outputs=merged)
        model.summary()
        return model

    def get_file_list(self, bin, num_peaks, cline, inclusive=False):
        dirname = os.path.join(self.root, 'trn_data/{}bp_{}peaks'.format(bin, num_peaks))
        # exclude cline
        if inclusive:
            flist = sorted([os.path.join(dirname, '_'.join(x.split('_')[:4])) for x in os.listdir(dirname) if
                            cline in x and '_lab.csv' in x])
        else:
            flist = sorted([os.path.join(dirname, '_'.join(x.split('_')[:4])) for x in os.listdir(dirname) if
                            cline not in x and '_lab.csv' in x])

        n = len(flist)
        m = int(n * 0.1)
        return flist[:m], flist[m:-m], flist[-m:]

    def trainAndValidate(self, test='K562', net_model='cnn', bin=20, num_peaks=2):
        val_flist, trn_flist, tst_flist = self.get_file_list(bin, num_peaks, test)
        batch_size = 1

        trn_batch_generator = Generator(trn_flist, batch_size, num_peaks)
        val_batch_generator = Generator(val_flist, batch_size, num_peaks)
        tst_batch_generator = Generator(tst_flist, batch_size, num_peaks)

        seq_in = Input((1000, 5))
        peak_in = Input((50, num_peaks))
        model = self.model(peak_in, seq_in, net_model)

        sgd = SGD(lr=0.01, momentum=0.5, decay=0.0, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        history = model.fit_generator(generator=trn_batch_generator,
                                      epochs=15,
                                      verbose=1,
                                      validation_data=val_batch_generator)

        model.save(os.path.join(self.root, 'model/{}bp_{}peaks/{}/model_{}.vgg'.format(bin, num_peaks, net_model, test)))
        pd.DataFrame(history.history).to_csv(os.path.join(self.root, 'model/{}bp_{}peaks/{}/model_{}.csv'.format(bin, num_peaks, net_model, test)), index=False)

        model = load_model(os.path.join(self.root, 'model/{}bp_{}peaks/{}/model_{}.vgg'.format(bin, num_peaks, net_model, test)), compile=False)

        true_label, pred_label = [], []
        prob = []
        for i, tb in enumerate(tst_batch_generator):
            print(i)
            data, label = tb
            pred = model.predict(data)
            prob.append(pred)
            pred_label.append(pred.argmax(axis=1))
            true_label.append(label.argmax(axis=1))

        prob = np.concatenate(prob)
        pred_label = np.concatenate(pred_label)
        true_label = np.concatenate(true_label)

        pd.DataFrame(metrics.confusion_matrix(true_label, pred_label)).to_csv(os.path.join(self.root, 'model/{}bp_{}peaks/{}/test_classification_cmat_{}.csv'.format(bin, num_peaks, net_model, test)))
        with open(os.path.join(self.root, 'model/{}bp_{}peaks/{}/test_classification_report_{}.txt'.format(bin, num_peaks, net_model, test)), 'wt') as f:
            f.write(metrics.classification_report(true_label, pred_label, digits=4))

        ridx, cidx = np.where(prob > 0.8)
        len_tl = len(true_label)
        true_label = true_label[ridx]
        pred_label = pred_label[ridx]
        len_tl_wt = len(true_label)

        print(len_tl_wt / len_tl)
        pd.DataFrame(metrics.confusion_matrix(true_label, pred_label)).to_csv(os.path.join(self.root, 'model/{}bp_{}peaks/{}/test_classification_cmat_{}_thres.csv'.format(bin, num_peaks, net_model, test)))
        with open(os.path.join(self.root, 'model/{}bp_{}peaks/{}/test_classification_report_{}_thres.txt'.format(bin, num_peaks, net_model, test)), 'wt') as f:
            f.write(metrics.classification_report(true_label, pred_label, digits=4))

    def sequential_train(self, test='K562', net_model='cnn', bin=20, num_peaks=2):
        val_flist, trn_flist, tst_flist = self.get_file_list(bin, num_peaks, test)
        batch_size = 1

        print(trn_flist)
        print(val_flist)

        peak_trn_batch_generator = Generator(trn_flist, batch_size, num_peaks, type='peak')
        peak_val_batch_generator = Generator(val_flist, batch_size, num_peaks, type='peak')
        peak_tst_batch_generator = Generator(tst_flist, batch_size, num_peaks, type='peak')

        seq_trn_batch_generator = Generator(trn_flist, batch_size, num_peaks, type='sequence')
        seq_val_batch_generator = Generator(val_flist, batch_size, num_peaks, type='sequence')
        seq_tst_batch_generator = Generator(tst_flist, batch_size, num_peaks, type='sequence')

        seq_in = Input((1000, 5))
        peak_in = Input((50, num_peaks))
        pmodel, smodel = self.sequential_model(peak_in, seq_in)

        sgd = SGD(lr=0.01, momentum=0.5, decay=0.0, nesterov=False)

        smodel.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        seq_history = smodel.fit_generator(generator=seq_trn_batch_generator,
                                           epochs=15,
                                           verbose=1,
                                           validation_data=seq_val_batch_generator,
                                           max_queue_size=32)
        smodel.save(os.path.join(self.root, 'model/{}bp_{}peaks/{}/seq_model_{}.vgg'.format(bin, num_peaks, net_model, test)))

        smodel = load_model(os.path.join(self.root, 'model/{}bp_{}peaks/{}/seq_model_{}.vgg'.format(bin, num_peaks, net_model, test)), compile=False)

        pmodel.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        peak_history = pmodel.fit_generator(generator=peak_trn_batch_generator,
                                            epochs=15,
                                            verbose=1,
                                            validation_data=peak_val_batch_generator,
                                            max_queue_size=32)
        pmodel.save(os.path.join(self.root, 'model/{}bp_{}peaks/{}/peak_model_{}.vgg'.format(bin, num_peaks, net_model, test)))

        pd.DataFrame(peak_history.history).to_csv(os.path.join(self.root, 'model/{}bp_{}peaks/{}/peak_model_{}.csv'.format(bin, num_peaks, net_model, test)), index=False)
        pd.DataFrame(seq_history.history).to_csv(os.path.join(self.root, 'model/{}bp_{}peaks/{}/seq_model_{}.csv'.format(bin, num_peaks, net_model, test)), index=False)

        pmodel = load_model(os.path.join(self.root, 'model/{}bp_{}peaks/{}/peak_model_{}.vgg'.format(bin, num_peaks, net_model, test)), compile=False)

        peak_pred_res, seq_pred_res = [], []
        true_label = []
        for peak, sequence in zip(peak_tst_batch_generator, seq_tst_batch_generator):
            peak = peak[0]
            sequence, label = sequence
            true_label.append(label.argmax(axis=1))

            peak_pred = pmodel.predict(peak)
            seq_pred = smodel.predict(sequence)

            peak_pred_res.append(peak_pred)
            seq_pred_res.append(seq_pred)

        peak_pred_res = np.concatenate(peak_pred_res)
        seq_pred_res = np.concatenate(seq_pred_res)
        true_label = np.concatenate(true_label)

        pred_label = []
        prob = np.zeros_like(peak_pred_res)
        i = 0
        for ppr, spr in zip(peak_pred_res, seq_pred_res):
            if ppr.argmax() >= spr.argmax():
                pred_label.append(ppr.argmax())
                prob[i, :] = ppr
            else:
                pred_label.append(spr.argmax())
                prob[i, :] = spr
            i += 1

        pred_label = np.array(pred_label)

        pd.DataFrame(metrics.confusion_matrix(true_label, pred_label)).to_csv(os.path.join(self.root, 'model/{}bp_{}peaks/{}/seq_test_classification_cmat_{}.txt'.format(bin, num_peaks, net_model, test)))
        with open(os.path.join(self.root, 'model/{}bp_{}peaks/{}/seq_test_classification_report_{}.txt'.format(bin, num_peaks, net_model, test)), 'wt') as f:
            f.write((metrics.classification_report(true_label, pred_label, digits=4)))

        ridx, cidx = np.where(prob > 0.8)

        true_label = true_label[ridx]
        pred_label = pred_label[ridx]
        pd.DataFrame(metrics.confusion_matrix(true_label, pred_label)).to_csv(os.path.join(self.root, 'model/{}bp_{}peaks/{}/seq_test_classification_cmat_{}_thres.txt'.format(bin, num_peaks, net_model, test)))
        with open(os.path.join(self.root, 'model/{}bp_{}peaks/{}/seq_test_classification_report_{}_thres.txt'.format(bin, num_peaks, net_model, test)), 'wt') as f:
            f.write((metrics.classification_report(true_label, pred_label, digits=4)))

    def crossvalidation(self, bin, i, net_model='cnn', cline='K562', num_peaks=2):
        val_flist, trn_flist, tst_flist = self.get_file_list(bin, num_peaks, cline)
        batch_size = 1

        flist = val_flist + trn_flist + tst_flist
        n = len(flist)
        m = int(n * 0.1)

        val_flist = flist[i * m: (i + 1) * m]
        if (i + 1) * m >= n:
            j = 0
        else:
            j = i
        tst_flist = flist[(j + 1) * m - 1: (j + 2) * m - 1]
        trn_flist = list(set(flist) - set(val_flist) - set(tst_flist))

        trn_batch_generator = Generator(trn_flist, batch_size, num_peaks)
        val_batch_generator = Generator(val_flist, batch_size, num_peaks)
        tst_batch_generator = Generator(tst_flist, batch_size, num_peaks)

        seq_in = Input((1000, 5))
        peak_in = Input((50, num_peaks))
        model = self.model(peak_in, seq_in, net_model)

        sgd = SGD(lr=0.01, momentum=0.5, decay=0.0, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        history = model.fit_generator(generator=trn_batch_generator,
                                      epochs=15,
                                      verbose=1,
                                      validation_data=val_batch_generator)

        model.save(os.path.join(self.root, 'model/{}bp_{}peaks/{}/model_{}{}.vgg'.format(bin, num_peaks, net_model, cline, i)))
        pd.DataFrame(history.history).to_csv(os.path.join(self.root, 'model/{}bp_{}peaks/{}/model_{}{}.csv'.format(bin, num_peaks, net_model, cline, i)), index=False)

        model = load_model(os.path.join(self.root, 'model/{}bp_{}peaks/{}/model_{}{}.vgg'.format(bin, num_peaks, net_model, cline, i)), compile=False)

        true_label, pred_label = [], []
        prob = []
        for tb in tst_batch_generator:
            data, label = tb
            pred = model.predict(data)
            prob.append(pred)
            pred_label.append(pred.argmax(axis=1))
            true_label.append(label.argmax(axis=1))

        prob = np.concatenate(prob)
        pred_label = np.concatenate(pred_label)
        true_label = np.concatenate(true_label)

        pd.DataFrame(metrics.confusion_matrix(true_label, pred_label)).to_csv(os.path.join(self.root, 'model/{}bp_{}peaks/{}/test_classification_cmat_{}{}.txt'.format(bin, num_peaks, net_model, cline, i)))
        with open(os.path.join(self.root, 'model/{}bp_{}peaks/{}/test_classification_report_{}{}.txt'.format(bin, num_peaks, net_model, cline, i)), 'wt') as f:
            f.write(metrics.classification_report(true_label, pred_label, digits=4))

        ridx, cidx = np.where(prob > 0.8)
        true_label = true_label[ridx]
        pred_label = pred_label[ridx]
        pd.DataFrame(metrics.confusion_matrix(true_label, pred_label)).to_csv(os.path.join(self.root, 'model/{}bp_{}peaks/{}/test_classification_cmat_{}_thres{}.txt'.format(bin, num_peaks, net_model, cline, i)))
        with open(os.path.join(self.root,
                               'model/{}bp_{}peaks/{}/test_classification_report_{}_thres{}.txt'.format(bin, num_peaks, net_model, cline, i)), 'wt') as f:
            f.write(metrics.classification_report(true_label, pred_label, digits=4))

    def test_unseen_cell_line(self, cline, network, bin, n):
        model = load_model(os.path.join(self.root, 'model/{}bp_{}peaks/{}/model_{}.vgg'.format(bin, n, network, cline)),
                           compile=False)

        val_flist, trn_flist, tst_flist = self.get_file_list(bin, n, cline, inclusive=True)
        trn_batch_generator = Generator(val_flist + trn_flist + tst_flist, 1, n, loc=True)

        test_data = {'peak': [], 'seq': [], 'lab': [], 'loc': []}
        for trn in trn_batch_generator:
            peak, sequence = trn[0]
            label, loc = trn[1]
            test_data['peak'].append(peak)
            test_data['seq'].append(sequence)
            test_data['lab'].append(label)
            test_data['loc'].append(loc)

        for key in test_data.keys():
            test_data[key] = np.concatenate(test_data[key])

        prob = model.predict([test_data['peak'], test_data['seq']])
        pred_label = prob.argmax(axis=1)

        true_label = test_data['lab'].argmax(axis=1)

        df_loc = pd.Series(test_data['loc']).str.split(';', expand=True)
        df_loc.columns = ['chromosome', 'tss', 'strand']
        df_loc['tss'] = df_loc['tss'].astype(int)
        df_loc.index = test_data['loc']

        pd.DataFrame(metrics.confusion_matrix(true_label, pred_label)).to_csv(os.path.join(self.root, 'model/{}bp_{}peaks/unseen_classification_cmat_{}.csv'.format(bin, num_peaks, cline)))

        # without threshold
        with open(os.path.join(self.root, 'model/{}bp_{}peaks/{}/unseen_classification_report_{}.txt'.format(bin, n, network, cline)), 'wt') as f:
            f.write(metrics.classification_report(true_label, pred_label, digits=4))

        len_wo_tr = len(pred_label)

        # with threshold
        ridx, cidx = np.where(prob > 0.8)
        prob = prob.max(axis=1)

        pd.DataFrame(metrics.confusion_matrix(true_label[ridx], pred_label[ridx])).to_csv(os.path.join(self.root, 'model/{}bp_{}peaks/unseen_classification_cmat_{}.csv'.format(bin, num_peaks, cline)))

        with open(os.path.join(self.root, 'model/{}bp_{}peaks/{}/unseen_classification_report_{}_thres.txt'.format(bin, n, network, cline)), 'wt') as f:
            f.write(metrics.classification_report(true_label[ridx], pred_label[ridx], digits=4))

        len_w_tr = len(pred_label[ridx])
        print(len_w_tr / len_wo_tr)

        distances = {}
        distribution = {}
        for res, truth, p, idx in zip(pred_label, true_label, prob, df_loc.index):
            distance = abs(res - truth) * 20
            distances[distance] = distances.get(distance, 0) + 1
            df_loc.loc[idx, 'probability'] = p
            df_loc.loc[idx, 'start'] = df_loc.loc[idx, 'tss'] - 500 + res * bin
            df_loc.loc[idx, 'end'] = df_loc.loc[idx, 'start'] + bin
            if distance not in distribution:
                distribution[distance] = [p]
            else:
                distribution[distance].append(p)
        distances = pd.Series(distances).sort_index() / pred_label.shape[0]
        df_loc.to_csv(os.path.join(self.root, 'pred/{}bp_{}peaks/{}'.format(bin, n, network), 'pred_tss_{}.csv'.format(cline)), index=False)

        plt.subplot(211)
        plt.plot(distances.index, distances)
        plt.grid()
        plt.subplot(212)
        plt.plot(distances.index, distances.cumsum())
        plt.grid()
        plt.savefig(os.path.join(self.root, 'figures/{}bp_{}peaks/{}/pred_res_{}.png'.format(bin, n, network, cline)))
        plt.close()

        distance = sorted(distribution.keys())
        plt.figure(figsize=(12, 10))
        for i, dist in enumerate(distance):
            prob = np.array(distribution[dist])
            if i > 8:
                continue
            plt.subplot(3, 3, i + 1)

            hist, bin_edges = np.histogram(prob, bins=10)
            hist = hist / hist.sum()
            plt.plot(bin_edges[1:], hist)
            plt.grid()
            m = prob.mean()
            s = prob.std()
            plt.title('[{}] mean: {:0.2f}, std: {:0.2f}'.format(dist, m, s))
            midx = np.abs(hist - m).argmin()
            plt.scatter(bin_edges[midx + 1], hist[midx], linewidths=2, marker='^', color='r')
            plt.savefig(os.path.join(self.root, 'figures/{}bp_{}peaks/{}/prob_distributions_{}.png'.format(bin, n, network, cline)))
        plt.close()


if __name__ == '__main__':
    root = "D:/DmiRT"
    net = Network(root)
    num_peaks = 2
    cell_lines = ['A549', 'GM12878', 'HES', 'HelaS3', 'HepG2', 'K562', 'MCF7']

    for cline in cell_lines:
        for bin in [20, 100]:
            net.trainAndValidate(cline, 'cnn', bin, num_peaks)
            net.sequential_train(cline, 'cnn', bin, num_peaks)
            net.test_unseen_cell_line(cline, 'cnn', bin, num_peaks)

    for i in range(10):
        net.crossvalidation(20, i, 'cnn', 'K562', num_peaks)
