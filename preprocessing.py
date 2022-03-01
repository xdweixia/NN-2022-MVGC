import numpy as np
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adjs):
    numView = len(adjs)
    adjs_normarlized = []
    for v in range(numView):
        adj = sp.coo_matrix(adjs[v])
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).toarray()
        adjs_normarlized.append(adj_normalized.tolist())
    return np.array(adjs_normarlized)


def construct_feed_dict(features1, features2, adj1, adj2, adj_orig1, adj_orig2, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features1']: features1})
    feed_dict.update({placeholders['features2']: features2})
    feed_dict.update({placeholders['adjs1']: adj1})
    feed_dict.update({placeholders['adjs2']: adj2})
    feed_dict.update({placeholders['adjs_orig1']: adj_orig1})
    feed_dict.update({placeholders['adjs_orig2']: adj_orig2})
    return feed_dict