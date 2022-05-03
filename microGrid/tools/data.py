import numpy as np
import joblib
import csv

class Data:
    def read_npy(self, path):
        return np.load(path)

    def save_npy(self, array, path):
        return np.save(path, array)

    def read_object(self, path):
        joblib.load(path)

    def save_object(self, obj, path):
        joblib.dump(obj, path, compress=True)

def read_csv(f, sep=";", skip_line=0, end_line=None):
    f = open(f)
    res = None
    csv_file = csv.reader(f, delimiter=sep)
    for i, l in enumerate(csv_file):
        if i<skip_line:
            continue
        if end_line is not None and i > end_line:
            break
        if res is None:
            res= [[] for _ in l]
        for i in range(len(l)):
            res[i].append(l[i])
    return res

def cumul_data(mat_csv, column_need, column_uniqu=None, skip_line=0):
    res = []
    uniqu = dict()
    for i, val in enumerate(mat_csv[column_need]):
        if i<skip_line:
            continue
        if not bool(val.strip()):
            val=0
        if column_uniqu is None:
            res.append(float(val))
        elif mat_csv[column_uniqu][i] in uniqu:
            res[uniqu[mat_csv[column_uniqu][i]]] += float(val)
        else:
            uniqu[mat_csv[column_uniqu][i]] = len(res)
            res.append(float(val))
    return res

def add_n_by_n(l, n):
    res = []
    for i, val in enumerate(l):
        if i % n == 0:
            res.append(val)
        else:
            res[-1] += val
    return res