import numpy as np
import scipy.io as sio

def load_data(dataset):
    # load the data: x, tx, allx, graph
    global dataset_path, rownetworks, truelabels, truefeatures1, truefeatures2
    if dataset == "ACM":
        dataset_path = "ACM.mat"
    elif dataset == "DBLP":
        dataset_path = "DBLP.mat"
    data = sio.loadmat(dataset_path)

    if dataset == "ACM":
        truelabels, truefeatures1, truefeatures2= data['Y'], data['X1'].astype(float), data['X2'].astype(float)
        rownetworks = np.array([(data['A1']).tolist(), (data['A2']).tolist()])
    elif dataset == "DBLP":
        truelabels, truefeatures1 = data['label'], data['features'].astype(float)
        rownetworks = np.array([(data['net_APA']).tolist(), (data['net_APCPA']).tolist(), (data['net_APTPA']).tolist()])

    numView = rownetworks.shape[0]
    y = truelabels
    return np.array(rownetworks), numView, truefeatures1, truefeatures2, truelabels
