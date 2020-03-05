# Useful code snippets
An eclectic collection of useful code snippets with usage examples. This will grow over time as I add various bits of useful code I've written over the years. As it grows, I'll probably sort each snippet into categories.

- [Converting a Pandas DataFrame to a dictionary -- 03/04/2020](#converting-a-pandas-dataframe-to-a-dictionary)
- [Running PCA on scipy sparse matrices -- 03/04/2020](#running-pca-on-scipy-sparse-matrices)
- [Converting a scipy sparse adjacency matrix to a k-nearest neighbor graph -- 03/05/2020](#converting-a-scipy-sparse-adjacency-matrix-to-a-k-nearest-neighbor-graph)
- [Splitting and modifying arrays of strings -- 03/05/2020](#splitting-and-modifying-arrays-of-strings)

## Converting a Pandas DataFrame to a dictionary
This is much faster than the built in `to_dict` function in Pandas DataFrame. Also, Pandas DataFrames do not handle cases where the same key may appear multiple times with different values. In my implementation, all values that are associated with a particular key are concatenated into an array.

Parameters:
  - `DF` is the input DataFrame.
  - `key_key` is the column ID that will be the dictionary key. If `None`, the dictionary key values will be the index of `DF`
  - `val_key` is a list of column IDs that will be the dictionary values. If it is an empty list, all columns will be used.

Example:
```python
data = np.array([['A',0,1,2],['A',3,6,7],['A',8,9,10],
                ['B',3,4,5],['B','hello','world',5],['B',3,'foo',5],
                ['C',6,7,8]])
DF = pd.DataFrame(data = data,columns=['W','X','Y','Z'])

res1 = df_to_dict(DF,key_key='W',val_key=['Y','Z'])
res2 = df_to_dict(DF,key_key='W',val_key=['X','Y'])
print(DF)
print(res1)
print(res2)
```
Function:
```python
import numpy as np
def df_to_dict(DF,key_key=None,val_key=[]):
    if key_key is None:
        index = list(DF.index)
    else:
        index = list(DF[key_key].values)

    if len(val_key) == 0:
        val_key = list(DF.columns)

    a=[]; b=[];
    for key in val_key:
        if key != key_key:
            a.extend(index)
            b.extend(list(DF[key].values))
    a=np.array(a); b=np.array(b);


    idx = np.argsort(a)
    a = a[idx]
    b = b[idx]
    bounds = np.where(a[:-1]!=a[1:])[0]+1
    bounds = np.append(np.append(0,bounds),a.size)
    bounds_left=bounds[:-1]
    bounds_right=bounds[1:]
    slists = [b[bounds_left[i]:bounds_right[i]]
                    for i in range(bounds_left.size)]
    d = dict(zip(np.unique(a),slists))
    return d
```

## Running PCA on scipy sparse matrices

This makes use of the `LinearOperator` class to create customized dot products that can be utilized by scipy sparse matrices. This allows us to incorporate implicit mean centering into the sparse SVD algorithms provided by `scipy.sparse.linalg.svds`.

Parameters:
  - `X` -- Input data (scipy.sparse.csr_matrix or scipy.sparse.csc_matrix)
  - `npcs` -- Number of principal components to use
  - `solver` -- For now, can be either `'arpack'` or `'lobpcg'`.
  - `mu` -- If you've precomputed the feature means of `X`, you can pass them in here.
  - `random_state` -- The random seed that can be set for reproducibility (integer or `numpy.random.RandomState`)
    
Example:
```python
#given a sparse matrix X
res = pca_with_sparse(X,50)
```
Function:
```python
import numpy as np
from scipy import sparse
from sklearn.utils.extmath import svd_flip
from sklearn.utils import check_array, check_random_state

def pca_with_sparse(X, npcs, solver='arpack', mu=None, random_state=None):
    random_state = check_random_state(random_state)
    np.random.set_state(random_state.get_state())
    random_init = np.random.rand(np.min(X.shape))
    X = check_array(X, accept_sparse=['csr', 'csc'])

    if mu is None:
        mu = X.mean(0).A.flatten()[None, :]
    mdot = mu.dot
    mmat = mdot
    mhdot = mu.T.dot
    mhmat = mu.T.dot
    Xdot = X.dot
    Xmat = Xdot
    XHdot = X.T.conj().dot
    XHmat = XHdot
    ones = np.ones(X.shape[0])[None, :].dot

    def matvec(x):
        return Xdot(x) - mdot(x)

    def matmat(x):
        return Xmat(x) - mmat(x)

    def rmatvec(x):
        return XHdot(x) - mhdot(ones(x))

    def rmatmat(x):
        return XHmat(x) - mhmat(ones(x))

    XL = sparse.linalg.LinearOperator(
        matvec=matvec,
        dtype=X.dtype,
        matmat=matmat,
        shape=X.shape,
        rmatvec=rmatvec,
        rmatmat=rmatmat,
    )

    u, s, v = sparse.linalg.svds(XL, solver=solver, k=npcs, v0=random_init)
    u, v = svd_flip(u, v)
    idx = np.argsort(-s)
    v = v[idx, :]

    X_pca = (u * s)[:, idx]
    ev = s[idx] ** 2 / (X.shape[0] - 1)

    total_var = _get_mean_var(X)[1].sum()
    ev_ratio = ev / total_var

    output = {
        'X_pca': X_pca,
        'variance': ev,
        'variance_ratio': ev_ratio,
        'components': v,
    }
    return output
```
## Converting a scipy sparse adjacency matrix to a k-nearest neighbor graph
Given a large, `scipy.sparse` adjacency matrix (representing a graph), we want to convert it to a k-nearest neighbor graph without needing to densify the data. I also provide a function (`sparse_knn_ks`) to convert the graph to a k-nearest neighbor graph with variable k.

Parameters:
- `D` -- your `scipy.sparse` adjacency matrix
- `k` -- the number of nearest neighbors to keep
Example:
```
#given a sparse adjacency matrix D, find 15 nearest neighbors
knnm = sparse_knn(D,15)

#given a sparse adjacency matrix D and a vector of #nearest neighbors ks, find k_i nearest neighbors for each sample i
#ks = [15,14,20,...,25,30,10]
knnm2 = sparse_knn_ks(D,ks)
```
Functions:
```python
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

## Splitting and modifying arrays of strings
Given a vector of strings, we want to be able extract a specific portion of each string. I use this function a lot.


Parameters/Examples:
- `substr(['A_foo','A_hello','B_world'],s='_',ix = 0)` splits each string by `_` and returns the first substring (`ix=0`), yielding `array(['A','A','B'])`. 
- `substr(['A_foo','A_hello','B_world'],s='_',ix = 1)` returns `array(['foo','hello','world'])`.
- `substr(['A_foo','A_hello','B_world_x'],s='_')` returns a list of all possible splits: `[array(['A','A','B']),array(['foo','hello','world']),array(['','','x'])]`
- If `obj=True`, the numpy array returned will have `'object'` data type. Otherwise, the array will have unicode string data type. The `'object'` data type is extremely useful if you want to concatenate a string to an array of strings or two arrays of strings together in an element-wise fashion: 
```
a = ['A_1','B_2','C_3']
b = ['1_foo','2_hello','3_world']
c = substr(a,s='_',ix=0,obj=True)+'_'+substr(b,s='_',ix=1,obj=True)
print(c)
```
would print `['A_foo','B_hello','C_world']`. Note that the `'object'` dtype is unwieldy for large vectors, so make sure to transform it back to unicode datatype. An easy way of doing this is by casting to a list and then a numpy array: `c=np.array(list(c))`. 

Using the `substr` function, you can now easily strip unwanted string headers and add new information in a vectorized fashion.

Function:
```python
import numpy
def substr(x, s="_", ix=None,obj=False):
    m = []    
    if ix is not None:
        for i in range(len(x)):
            f = x[i].split(s)
            ix = min(len(f) - 1, ix)
            m.append(f[ix])
        return np.array(m).astype('object') if obj else np.array(m)
    else:
        ms = []
        ls = []
        for i in range(len(x)):
            f = x[i].split(s)
            m = []
            for ix in range(len(f)):
                m.append(f[ix])
            ms.append(m)
            ls.append(len(m))
        ml = max(ls)
        for i in range(len(ms)):
            ms[i].extend([""] * (ml - len(ms[i])))
            if ml - len(ms[i]) > 0:
                ms[i] = np.concatenate(ms[i])
        ms = np.vstack(ms)
        if obj:
            ms=ms.astype('object')
        MS = []
        for i in range(ms.shape[1]):
            MS.append(ms[:, i])
        return MS
```
