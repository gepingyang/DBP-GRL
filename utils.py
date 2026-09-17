import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset, CoraGraphDataset
from dgl.data import WikiCSDataset
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit



def get_normalized_adj_sparse(g, num_nodes):
    src, dst = g.edges()


    mask = src != dst
    src, dst = src[mask], dst[mask]


    row = torch.cat([src, dst])
    col = torch.cat([dst, src])
    indices = torch.stack([row, col])


    temp_adj = torch.sparse_coo_tensor(
        indices,
        torch.ones(indices.shape[1], device=indices.device),
        (num_nodes, num_nodes)
    ).coalesce()

    new_indices = temp_adj.indices()
    new_values = torch.ones(new_indices.shape[1], device=indices.device)


    deg = torch.sparse.sum(
        torch.sparse_coo_tensor(new_indices, new_values, (num_nodes, num_nodes)),
        dim=1
    ).to_dense()


    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0


    row, col = new_indices[0], new_indices[1]
    final_values = deg_inv_sqrt[row] * deg_inv_sqrt[col]


    adj = torch.sparse_coo_tensor(
        indices=new_indices,
        values=final_values,
        size=(num_nodes, num_nodes)
    ).coalesce()

    return adj


def get_normalized_adj_from_edge_index(edge_index, num_nodes):
    """Build D^-1/2 A D^-1/2 from a PyG ``edge_index`` tensor."""
    src, dst = edge_index
    mask = src != dst
    src, dst = src[mask], dst[mask]

    indices = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ])
    adj = torch.sparse_coo_tensor(
        indices,
        torch.ones(indices.size(1), device=indices.device),
        (num_nodes, num_nodes),
    ).coalesce()

    row, col = adj.indices()
    degree = torch.sparse.sum(adj, dim=1).to_dense()
    degree_inv_sqrt = degree.pow(-0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0

    values = degree_inv_sqrt[row] * degree_inv_sqrt[col]
    return torch.sparse_coo_tensor(
        adj.indices(), values, adj.size(), device=values.device
    ).coalesce()


def _get_dataset(name):
    """Return the DGL dataset used by both node and link loaders."""
    name = name.lower()
    if name == 'wikics':
        return WikiCSDataset()
    if name == 'photo':
        return AmazonCoBuyPhotoDataset()
    if name == 'computer':
        return AmazonCoBuyComputerDataset()
    if name == 'co_physics':
        return CoauthorPhysicsDataset()
    if name == 'co_cs':
        return CoauthorCSDataset()
    if name == 'cora':

        return CoraGraphDataset()
    raise NotImplementedError(f"Unexpected Dataset: {name}")



def load(name):
    dataset = _get_dataset(name)



    graph = dataset[0]
    labels = graph.ndata.pop('label')
    N = graph.number_of_nodes()
    train_ratio = 0.1
    val_ratio = 0.1
    test_ratio = 0.8




    train_num = int(N * train_ratio)
    val_num = int(N * (train_ratio + val_ratio))

    idx = np.arange(N)
    np.random.shuffle(idx)

    train_idx = idx[:train_num]
    val_idx = idx[train_num:val_num]
    test_idx = idx[val_num:]

    train_idx = torch.tensor(train_idx)
    val_idx = torch.tensor(val_idx)
    test_idx = torch.tensor(test_idx)


    g = graph
    adj = get_normalized_adj_sparse(g, g.num_nodes())



    feat = g.ndata['feat']
    if name=='co_cs':
        feat = pca(feat)
    if name == 'co_physics':
        feat = pca(feat)
    # if name == 'cora':
    #      feat = pca(feat)

    num_class = dataset.num_classes


    return adj, feat, labels, num_class, train_idx, val_idx, test_idx


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret

def evaluate_classification(embeds, num_class, label, lr, wd, train_idx, val_idx, test_idx):
    for run in range(0, 1):
        train_idx_tmp = train_idx
        val_idx_tmp = val_idx
        test_idx_tmp = test_idx
        train_embs = embeds[train_idx_tmp]
        val_embs = embeds[val_idx_tmp]
        test_embs = embeds[test_idx_tmp]

        train_labels = label[train_idx_tmp]
        val_labels = label[val_idx_tmp]
        test_labels = label[test_idx_tmp]

        ''' Linear Evaluation '''
        logreg = LogReg(train_embs.shape[1], num_class)
        opt = torch.optim.Adam(logreg.parameters(), lr=lr, weight_decay=wd)

        logreg = logreg.to('cuda:0')
        loss_fn = nn.CrossEntropyLoss()

        best_val_acc = 0
        eval_acc = 0

        for epoch in range(500 + 1):
            logreg.train()
            opt.zero_grad()
            logits = logreg(train_embs)
            preds = torch.argmax(logits, dim=1)

            train_acc = torch.sum(preds == train_labels).float() / train_labels.shape[0]

            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()

            logreg.eval()
            with torch.no_grad():
                val_logits = logreg(val_embs)
                test_logits = logreg(test_embs)

                val_preds = torch.argmax(val_logits, dim=1)
                test_preds = torch.argmax(test_logits, dim=1)

                val_acc = torch.sum(val_preds == val_labels).float() / val_labels.shape[0]
                test_acc = torch.sum(test_preds == test_labels).float() / test_labels.shape[0]

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    eval_acc = test_acc
                print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc,
                                                                                         test_acc))

    return train_acc.cpu().numpy(), best_val_acc.cpu().numpy(), eval_acc.cpu().numpy()




def load_index(name):
    if name == 'wikics':
        dataset = WikiCSDataset()
    elif name == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name == 'computer':
        dataset = AmazonCoBuyComputerDataset()
    elif name == 'co_physics':
        dataset = CoauthorPhysicsDataset()
    elif name == 'co_cs':
        dataset = CoauthorCSDataset()
    else:
        raise NotImplementedError("Unexpected Dataset")
    graph = dataset[0]
    N = graph.number_of_nodes()
    labels = graph.ndata.pop('label')
    train_ratio = 0.1
    val_ratio = 0.1
    test_ratio = 0.8


    train_num = int(N * train_ratio)
    val_num = int(N * (train_ratio + val_ratio))

    idx = np.arange(N)
    np.random.shuffle(idx)

    train_idx = idx[:train_num]
    val_idx = idx[train_num:val_num]
    test_idx = idx[val_num:]
    train_idx = torch.tensor(train_idx)
    val_idx = torch.tensor(val_idx)
    test_idx = torch.tensor(test_idx)


    return  train_idx, val_idx, test_idx

def pca(X, n_components=512):
    X = F.normalize(X)
    covariance_matrix = torch.mm(X.T, X) / (X.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    components = eigenvectors_sorted[:, :n_components]
    X_pca = torch.mm(X, components)
    return X_pca
