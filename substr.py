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
