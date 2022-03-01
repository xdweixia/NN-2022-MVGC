from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from constructor import get_placeholder, get_model, format_data, get_optimizer, update, Fin_update
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
import scipy.io as sio
import warnings
from munkres import Munkres
from metrics import clustering_metrics
from scipy.fftpack import fft

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
warnings.filterwarnings('ignore')

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS


def best_map(L1, L2):
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


class Clustering_Runner():
    def __init__(self, settings):
        print("Clustering on dataset: %s, model: %s, number of pre_iteration: %3d, number of fin_iteration: %3d" % (
            settings['data_name'], settings['model'], settings['pre_iterations'], settings['fin_iterations']))
        self.data_name = settings['data_name']
        self.pre_iteration = settings['pre_iterations']
        self.fin_iteration = settings['fin_iterations']
        self.model = settings['model']
        self.n_clusters = settings['clustering_num']

    def erun(self):
        global Coef
        model_str = self.model

        # formatted data
        feas = format_data(self.data_name)
        # print(feas['num_features1'],feas['num_features2'],feas['num_nodes'], self.n_clusters)
        X2 = feas['features1']
        X1 = feas['features2'] # fft(X2) # np.matmul(X2,X2.T)
        A = feas['adjs']
        A1 = A[1]
        A2 = A[1]
        PW = feas['pos_weights']
        PW1 = PW[1]
        PW2 = PW[1]
        # print(PW1, PW2)
        # print(A1.shape,type(A1))
        # print(A2.shape, type(A2))
        L = np.squeeze(feas['true_labels'])
        # print(L.shape, type(L))

        # Define placeholders
        placeholders = get_placeholder()

        # construct model
        MGCN_model = get_model(model_str, placeholders, feas['num_features1'], feas['num_nodes'], self.n_clusters)
        #
        # Optimizer
        opt = get_optimizer(model_str, MGCN_model, placeholders)
        #
        # Initialize session
        sess = tf.Session()
        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        sess.run(tf.global_variables_initializer())

        # Pre_Train model
        for pre_epoch in range(self.pre_iteration):
            Coef, Pre_Total, Recost, SEloss, consistent_loss, S_Regular = update(MGCN_model, opt, sess, X1, X2, A1, A2, A1, A2, PW1, PW2, placeholders)
            print("Pre_epoch: %d" % pre_epoch, "Pre_Total: %.2f" % Pre_Total, "Re_Loss: %.2f" % Recost, "SEloss: %.2f" % SEloss, "consistent_loss: %.2f" % consistent_loss,
                  "S_Regular: %.2f" % S_Regular)
        alpha = max(0.4 - (3 - 1) / 10 * 0.1, 0.1)
        Ncoef = 0.5 * (np.abs(Coef) + np.abs(Coef.T))
        commonZ = self.thrC(Ncoef, alpha)
        y_x, _ = self.post_proC(commonZ, 3, 10, 3.5)
        cm = clustering_metrics(L, y_x + 1)
        acc, f1_macro, precision_macro, nmi, adjscore, _ = cm.evaluationClusterModelFromLabel()
        # missrate_x = self.err_rate(L, y_x + 1)
        # acc_x = 1 - missrate_x
        print(
            '----------------------------------------------------------------------------------------------------------')
        print("Initial Clustering Results: ")
        print("acc: {:.8f}\t\tf_score: {:.8f}\t\tnmi: {:.8f}\t\tari: {:.8f}".
              format(acc, f1_macro, nmi, adjscore))
        print(
            '----------------------------------------------------------------------------------------------------------')

        # Fin-tune Clustering model
        s2_label_subjs = np.array(y_x)
        s2_label_subjs = s2_label_subjs - s2_label_subjs.min() + 1
        s2_label_subjs = np.squeeze(s2_label_subjs)
        one_hot_Label = self.get_one_hot_Label(s2_label_subjs)
        s2_Q = self.form_structure_matrix(s2_label_subjs, 3)
        s2_Theta = self.form_Theta(s2_Q)
        Y = y_x
        for fin_epoch in range(self.fin_iteration):
            s2_Coef, Fin_Total, cost, SEloss, consistent_loss, S_Regular, Cq_loss, dense_loss, center_loss, csd = \
                Fin_update(MGCN_model, opt, sess, X1,  X2,  A1, A2,  A1,   A2,  PW1,  PW2, one_hot_Label, s2_Theta, Y, placeholders)
            if fin_epoch % (5) == 0:
                # sio.savemat('C' + str(fin_epoch) + '.mat', {'C': s2_Coef})
                s2_label_subjs = np.array(Y)
                s2_label_subjs = s2_label_subjs - s2_label_subjs.min() + 1
                s2_label_subjs = np.squeeze(s2_label_subjs)
                one_hot_Label = self.get_one_hot_Label(s2_label_subjs)
                s2_Q = self.form_structure_matrix(s2_label_subjs, 3)
                s2_Theta = self.form_Theta(s2_Q)
                s2_Coef = self.thrC(s2_Coef, alpha)
                y_x, Soft_Q = self.post_proC(s2_Coef, 3, 10, 3.5)
                if len(np.unique(y_x)) != 3:
                    continue
                Y = best_map(Y + 1, y_x + 1) - 1
                Y = Y.astype(np.int)
                # s2_missrate_x = self.err_rate(L, Y + 1)
                # s2_acc_x = 1 - s2_missrate_x
                cm = clustering_metrics(L, Y + 1)
                acc, f1_macro, precision_macro, nmi, adjscore, _ = cm.evaluationClusterModelFromLabel()
                # s2_nmi_x = nmi(L, Y + 1)
                # s2_ari_x = ari(L, Y + 1)
                # s2_Fs_x = f_score(L, Y + 1, average='macro')
                print("Fin_epoch: %d" % fin_epoch, "Fin_Total: %.2f" % Fin_Total, "Re_Loss: %.2f" % cost,
                      "SEloss: %.2f" % SEloss, "consistent_loss: %.2f" % consistent_loss,
                      "S_Regular: %.2f" % S_Regular, "Cq_loss: %.2f" % Cq_loss,
                      "dense_loss: %.2f" % dense_loss,
                      "center_loss: %.2f" % center_loss,
                      "csd: %.2f" % csd)
                fh = open('Loss.txt', 'a')
                fh.write('Fin_epoch=%d, Fin_Total: %.2f, Re_Loss: %.2f, SEloss: %.2f, dense_loss: %.2f, csd: %.2f, consistent_loss: %.2f, Cq_loss: %.2f ' % (
                    fin_epoch, Fin_Total, cost, SEloss, dense_loss, csd, consistent_loss, Cq_loss))
                fh.write('\r\n')
                fh.flush()
                fh.close()
                print("-------------------------------------------------------------")
                print("Rearrange:", "\033[1;31;43m MVGC_Acc:%.4f \033[0m" % acc)
                print("Rearrange:", "\033[1;31;43m MVGC_Fsc:%.4f \033[0m" % f1_macro)
                print("Rearrange:", "\033[1;31;43m MVGC_Nmi:%.4f \033[0m" % nmi)
                print("Rearrange:", "\033[1;31;43m MVGC_Ari:%.4f \033[0m" % adjscore)

    def thrC(self, C, ro):
        if ro < 1:
            N = C.shape[1]
            Cp = np.zeros((N, N))
            S = np.abs(np.sort(-np.abs(C), axis=0))
            Ind = np.argsort(-np.abs(C), axis=0)
            for i in range(N):
                cL1 = np.sum(S[:, i]).astype(float)
                stop = False
                csum = 0
                t = 0
                while stop == False:
                    csum = csum + S[t, i]
                    if csum > ro * cL1:
                        stop = True
                        Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                    t = t + 1
        else:
            Cp = C
        return Cp

    def post_proC(self, C, K, d, alpha):
        # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
        C = 0.5 * (C + C.T)
        r = d * K + 1
        U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
        U = U[:, ::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis=1)
        Z = U.dot(U.T)
        Z = Z * (Z > 0)
        L = np.abs(Z ** alpha)
        L = L / L.max()
        L = 0.5 * (L + L.T)
        spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                              assign_labels='discretize')
        spectral.fit(L)
        grp = spectral.fit_predict(L) + 1
        uu, ss, vv = svds(L, k=K)
        return grp, uu

    def form_structure_matrix(self, idx, K):
        Q = np.zeros((len(idx), K))
        for i, j in enumerate(idx):
            Q[i, j - 1] = 1
        return Q

    def form_Theta(self, Q):
        Theta = np.zeros((Q.shape[0], Q.shape[0]))
        for i in range(Q.shape[0]):
            Qq = np.tile(Q[i], [Q.shape[0], 1])
            Theta[i, :] = 1 / 2 * np.sum(np.square(Q - Qq), 1)
        return Theta

    def get_one_hot_Label(self, Label):
        if Label.min() == 0:
            Label = Label
        else:
            Label = Label - 1

        Label = np.array(Label)
        n_class = 3
        n_sample = Label.shape[0]
        one_hot_Label = np.zeros((n_sample, n_class))
        for i, j in enumerate(Label):
            one_hot_Label[i, j] = 1

        return one_hot_Label

    def err_rate(self, gt_s, s):
        c_x = best_map(gt_s, s)
        err_x = np.sum(gt_s[:] != c_x[:])
        missrate = err_x.astype(float) / (gt_s.shape[0])
        return missrate
