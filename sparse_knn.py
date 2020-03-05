import numpy as np
def sparse_knn(D,k):
    D1=D.tocoo()
    idr = np.argsort(D1.row)
    D1.row[:]=D1.row[idr]
    D1.col[:]=D1.col[idr]
    D1.data[:]=D1.data[idr]

    _,ind = np.unique(D1.row,return_index=True)
    ind = np.append(ind,D1.data.size)
    for i in range(ind.size-1):
        idx = np.argsort(D1.data[ind[i]:ind[i+1]])
        if idx.size > k:
            idx = idx[:-k]
            D1.data[np.arange(ind[i],ind[i+1])[idx]]=0
    D1.eliminate_zeros()
    return D1

def sparse_knn_ks(D,ks):
    D1=D.tocoo()
    idr = np.argsort(D1.row)
    D1.row[:]=D1.row[idr]
    D1.col[:]=D1.col[idr]
    D1.data[:]=D1.data[idr]

    row,ind = np.unique(D1.row,return_index=True)
    ind = np.append(ind,D1.data.size)
    for i in range(ind.size-1):
        idx = np.argsort(D1.data[ind[i]:ind[i+1]])
        k = ks[row[i]]
        if idx.size > k:
            if k != 0:
                idx = idx[:-k]
            else:
                idx = idx
            D1.data[np.arange(ind[i],ind[i+1])[idx]]=0
    D1.eliminate_zeros()
    return D1
```