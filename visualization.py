import numpy as np
import innvestigate.utils
from keras.layers import Input, Dense, concatenate, BatchNormalization, Flatten, Conv1D, MaxPooling1D, Dropout
from keras.models import Model
import matplotlib.pyplot as plt
from network import Generator, Network
import os


class Evaluation:
    def __init__(self, root):
        self.root = root

    def visualization(self, cline):
        seq_in = Input((1000, 5))
        peak_in = Input((50, 2))

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

        peak = Flatten()(peak)
        seq = Flatten()(seq)

        merged = concatenate([peak, seq])
        s = 4096
        for i in range(2):
            merged = Dense(s, activation='relu')(merged)
            merged = Dropout(rate=0.5)(merged)
        merged = Dense(50, activation='softmax')(merged)

        model = Model(inputs=[peak_in, seq_in], outputs=merged)
        model.summary()

        model = innvestigate.utils.model_wo_softmax(model)
        analyzer = innvestigate.create_analyzer("deep_taylor", model)

        net = Network(self.root)
        val_flist, trn_flist, tst_flist = net.get_file_list(20, 2, cline)
        tst_batch_generator = Generator(tst_flist, 1, 2)

        peak, sequence, label = [], [], []
        for tb in tst_batch_generator:
            d, l = tb
            p, s = d
            peak.append(p)
            sequence.append(s)
            label.append(l)
        peak = np.concatenate(peak, axis=0)
        sequence = np.concatenate(sequence, axis=0)
        label = np.concatenate(label, axis=0)

        index = label.argmax(axis=1)
        idx = np.where(index == 29)[0]
        peak = peak[idx]

        sequence = np.delete(sequence, 4, 0)
        sequence = sequence[idx]
        label = label[idx]

        peak = peak[:500]
        sequence = sequence[:500]

        analysis = analyzer.analyze([peak, sequence])
        for i in range(2):
            analysis[i] = analysis[i].sum(axis=np.argmax(np.asarray(analysis[i].shape) == 3))

            # Plot
            plt.rcParams.update({'font.size': 18})
            if i > 0:
                fig = plt.figure(figsize=(12, 4))
                f, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

                dz = analysis[i].T[:4, :]
                dz /= dz.max()

                c = ax1.pcolormesh(dz, cmap='RdBu')
                fig.colorbar(c, ax=ax1)
                ax1.set_yticks(np.arange(4)+0.5)
                ax1.set_yticklabels(['A', 'G', 'C', 'T'])

                ax1.get_xaxis().set_visible(False)

                cons = analysis[i].T[4, :]
                cons /= cons.max()

                ax2.plot(np.arange(len(cons)), cons)
                ax2.set_ylabel('Normalized Conservation Score')
                ax2.get_xaxis().set_visible(False)

            else:
                for j in range(analysis[i].shape[1]):
                    analysis[i][:, j] /= np.max(np.abs(analysis[i][:, j]))
                    ax = plt.subplot(analysis[i].shape[1], 1, j+1)
                    ax.plot(np.arange(analysis[i].shape[0]), analysis[i][:, j])
                    ax.get_xaxis().set_visible(False)
            # plt.show()
            plt.savefig(os.path.join(self.root, 'figures', 'innvestigate_{}_{}.png'.format(cline, i)))
            plt.close()


if __name__ == '__main__':
    root = "D:/Bioinformatics"
    eval = Evaluation(root)
    bin = 20
    network = 'cnn'
    for cline in ['K562']:
        eval.visualization(cline)
