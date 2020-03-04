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