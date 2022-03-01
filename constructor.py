import tensorflow as tf
import numpy as np
from model import MultiEncoder
from optimizer import Optimizer
import scipy.sparse as sp
from input_data import load_data
import inspect
from preprocessing import preprocess_graph, construct_feed_dict
flags = tf.app.flags
FLAGS = flags.FLAGS


def get_placeholder():
    placeholders = {
        'features1': tf.placeholder(tf.float32),
        'features2': tf.placeholder(tf.float32),
        'adjs1': tf.placeholder(tf.float32),
        'adjs2': tf.placeholder(tf.float32),
        'adjs_orig1': tf.placeholder(tf.float32),
        'adjs_orig2': tf.placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'attn_drop': tf.placeholder_with_default(0., shape=()),
        'ffd_drop': tf.placeholder_with_default(0., shape=()),
        'pos_weights1': tf.placeholder(tf.float32),
        'pos_weights2': tf.placeholder(tf.float32),
        'fea_pos_weights1': tf.placeholder(tf.float32),
        'fea_pos_weights2': tf.placeholder(tf.float32),
        'norm': tf.placeholder(tf.float32),
        'PL': tf.placeholder(tf.float32, shape=(None, 3)),
        'Theta': tf.placeholder(tf.float32, [3025, 3025]),
        'Labels': tf.placeholder(tf.int32)
    }
    return placeholders


def get_model(model_str, placeholders, num_features, num_nodes, num_clusters):
    model = None
    if model_str == 'Main':
        model = MultiEncoder(placeholders, num_features, num_clusters)
    return model


def format_data(data_name):
    print("Current dataset", data_name)
    rownetworks, numView, features1, features2, truelabels = load_data(data_name)
    # print(rownetworks.shape,type(rownetworks))
    adjs_orig = []
    for v in range(numView):
        adj_orig = rownetworks[v]
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adjs_orig.append(adj_orig)
    adjs_label = rownetworks
    adjs_orig = np.array(adjs_orig)
    adjs = adjs_orig
    if FLAGS.features == 0:
        features1 = sp.identity(features1.shape[0])  # featureless
        features2 = sp.identity(features2.shape[0])  # featureless
    # Some preprocessing
    adjs_norm = preprocess_graph(adjs)
    # print(adjs_norm.shape, type(adjs_norm))
    num_nodes = adjs[0].shape[0]
    features1 = features1
    features2 = features2
    num_features1 = features1.shape[1]
    num_features2 = features2.shape[1]
    fea_pos_weights1 = float(features1.shape[0] * features1.shape[1] - features1.sum()) / features1.sum()
    fea_pos_weights2 = float(features2.shape[0] * features2.shape[1] - features2.sum()) / features2.sum()
    pos_weights = []
    norms = []
    for v in range(numView):
        pos_weight = float(adjs[v].shape[0] * adjs[v].shape[0] - adjs[v].sum()) / adjs[v].sum()
        norm = adjs[v].shape[0] * adjs[v].shape[0] / float((adjs[v].shape[0] * adjs[v].shape[0] - adjs[v].sum()) * 2)
        pos_weights.append(pos_weight)
        norms.append(norm)
    true_labels = truelabels
    feas = {'adjs': adjs_norm, 'adjs_label': adjs_label, 'num_features1': num_features1, 'num_features2': num_features2,
            'num_nodes': num_nodes, 'true_labels': true_labels, 'pos_weights': pos_weights, 'norms': np.array(norms),
            'adjs_norm': adjs_norm, 'features1': features1, 'features2': features2, 'fea_pos_weights1': fea_pos_weights1,
            'fea_pos_weights2': fea_pos_weights2, 'numView': numView}
    return feas


def get_optimizer(model_str, model, placeholders):
    global opt
    if model_str == 'Main':
        opt = Optimizer(preds=model.reconstructions1,
                        labels=placeholders['adjs_orig1'],
                        preds2=model.reconstructions2,
                        labels2=placeholders['adjs_orig2'],
                        Z1=model.z_mean1,
                        ZC1=model.ZC1,
                        Z2=model.z_mean2,
                        ZC2=model.ZC2,
                        Coef=model.coef,
                        SZ1=model.SZ1,
                        SZ2=model.SZ2,
                        PL=placeholders['PL'],
                        Theta=placeholders['Theta'],
                        Labels =placeholders['Labels'],
                        pos_weight1=placeholders['pos_weights1'],
                        pos_weight2=placeholders['pos_weights2'])
    return opt


def update(model, opt, sess, features1, features2, adj1, adj2, adj_orig1, adj_orig2, PW1, PW2, placeholders):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features1, features2, adj1, adj2, adj_orig1, adj_orig2, placeholders)
    feed_dict.update({placeholders['pos_weights1']: PW1})
    feed_dict.update({placeholders['pos_weights2']: PW2})
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # print(len(feed_dict))
    _, Pre_Total, cost, SEloss, consistent_loss, S_Regular= sess.run([opt.Pre_opt_op, opt.loss, opt.cost, opt.SEloss, opt.consistent_loss, opt.S_Regular], feed_dict=feed_dict)
    Coef = sess.run(model.coef, feed_dict=feed_dict)
    return Coef, Pre_Total, cost, SEloss, consistent_loss, S_Regular


def Fin_update(model, opt, sess, features1, features2, adj1, adj2, adj_orig1, adj_orig2, PW1, PW2, PL, Theta, Labels, placeholders):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features1, features2, adj1, adj2, adj_orig1, adj_orig2, placeholders)
    feed_dict.update({placeholders['pos_weights1']: PW1})
    feed_dict.update({placeholders['pos_weights2']: PW2})
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['PL']: PL})
    feed_dict.update({placeholders['Theta']: Theta})
    feed_dict.update({placeholders['Labels']: Labels})
    _, Fin_Total, cost, SEloss, consistent_loss, S_Regular, Cq_loss, dense_loss, center_loss, csd = sess.run([opt.Fin_opt_op, opt.Total_Loss, opt.cost, opt.SEloss, opt.consistent_loss, opt.S_Regular,
                                                                       opt.Cq_loss, opt.dense_loss, opt.center_loss, opt.csd], feed_dict=feed_dict)
    Coef = sess.run(model.coef, feed_dict=feed_dict)
    return Coef, Fin_Total, cost, SEloss, consistent_loss, S_Regular, Cq_loss, dense_loss, center_loss, csd

