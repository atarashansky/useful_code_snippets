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