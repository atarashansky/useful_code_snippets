# Useful code snippets
A collection of useful code snippets with usage examples. This will grow over time as I add various bits of code.

[Converting a Pandas DataFrame to a dictionary](##Converting-a-Pandas-DataFrame-to-a-dictionary4)

## Converting a Pandas DataFrame to a dictionary
This is much faster than the built in `to_dict` function in Pandas DataFrame. Also, Pandas DataFrames do not handle cases where the same key may appear multiple times with different values. In my implementation, all values that are associated with a particular key are concatenated into an array.

Parameters:
  `DF` is the input DataFrame.
  `key_key` is the column ID that will be the dictionary key. If `None`, the dictionary key values will be the index of `DF`
  `val_key` is a list of column IDs that will be the dictionary values. If it is an empty list, all columns will be used.

Example:
```
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

## Converting a Pandas DataFrame to a dictionary
This is much faster than the built in `to_dict` function in Pandas DataFrame. Also, Pandas DataFrames do not handle cases where the same key may appear multiple times with different values. In my implementation, all values that are associated with a particular key are concatenated into an array.

Parameters:
  `DF` is the input DataFrame.
  `key_key` is the column ID that will be the dictionary key. If `None`, the dictionary key values will be the index of `DF`
  `val_key` is a list of column IDs that will be the dictionary values. If it is an empty list, all columns will be used.

Example:
```
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

## Converting a Pandas DataFrame to a dictionary
This is much faster than the built in `to_dict` function in Pandas DataFrame. Also, Pandas DataFrames do not handle cases where the same key may appear multiple times with different values. In my implementation, all values that are associated with a particular key are concatenated into an array.

Parameters:
  `DF` is the input DataFrame.
  `key_key` is the column ID that will be the dictionary key. If `None`, the dictionary key values will be the index of `DF`
  `val_key` is a list of column IDs that will be the dictionary values. If it is an empty list, all columns will be used.

Example:
```
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

## Converting a Pandas DataFrame to a dictionary4
This is much faster than the built in `to_dict` function in Pandas DataFrame. Also, Pandas DataFrames do not handle cases where the same key may appear multiple times with different values. In my implementation, all values that are associated with a particular key are concatenated into an array.

Parameters:
  `DF` is the input DataFrame.
  `key_key` is the column ID that will be the dictionary key. If `None`, the dictionary key values will be the index of `DF`
  `val_key` is a list of column IDs that will be the dictionary values. If it is an empty list, all columns will be used.

Example:
```
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
