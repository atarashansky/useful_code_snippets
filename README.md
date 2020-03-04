# Useful code snippets
A collection of useful code snippets with usage examples. This will grow over time as I add various bits of code.

- [Converting a Pandas DataFrame to a dictionary](#converting-a-pandas-dataframe-to-a-dictionary)
- [Running PCA on scipy sparse matrices](#running-pca-on-scipy-sparse-matrices)

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
