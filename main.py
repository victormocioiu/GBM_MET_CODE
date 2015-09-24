__author__ = 'Sunny'
from collections import OrderedDict
from itertools import izip, count
import os
import pickle

import numpy as np
from sklearn.cross_validation import StratifiedKFold, LeaveOneOut


def getData(conditions, condition_paths, echo):
    """Reads in memory all the data needed for the Glioblastoma vs Metastases paper.

	This function could be used for any kind of pickled data that holds a dictionary.
	It might not be the fastest function or the most pythonic, but I wrote it to get familiar with
	zip/izip - generators and the 'with' keyword.

	Args:
		conditions: Iterable containing the conditions(String) to be investigated
		condition_paths: Iterable containinf the paths to the conditions to be investigated
		echo: String  ('LONG' or 'SHORT') corresponding to the TE of the acquisition

	Returns:
		dataset: Nested OrderedDict where every primary key is a patient. Every patient dict has the following struct-
		labels : numpy 1D-array - the label (condition) for each patient
	"""
    extension = 'pkl'
    y = []
    dataset = OrderedDict()
    for index, condition, path in izip(count(), conditions, condition_paths):
        with open(os.path.join(path, condition + '_' + echo + '.' + extension), 'rb')as condition_file:
            data = pickle.load(condition_file)
            y.append(np.zeros(len(data) + index))
        dataset.update(data)
    return dataset, np.hstack(y)


if __name__ == "__main__":
    condit = [r'GBM', r'MET']
    paths = [r'C:\\Doctorat\\SOURCES\\BERN\\', r'C:\\Doctorat\\SOURCES\\BERN\\']
    echos = 'LONG'
    ## read in data
    dataset, y = getData(condit, paths, echos)
    ## funky format it
    X = []
    for patient in dataset.iterkeys():
        X.append(patient)
    X = np.array(X)
    skf = LeaveOneOut(X.shape[0])
    ## process in parallel

    ##analyze data
