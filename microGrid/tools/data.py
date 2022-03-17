import numpy as np
import joblib

class Data:
    def read_npy(self, path):
        return np.load(path)

    def save_npy(self, array,path):
        return np.save(path, array)

    def read_object(self, path):
        joblib.load(path)

    def save_object(self, obj, path):
        joblib.dump(obj, path, compress=True)
