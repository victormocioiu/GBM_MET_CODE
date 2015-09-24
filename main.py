__author__ = 'Sunny'
from collections import OrderedDict
from itertools import izip, count
import os
import pickle

import numpy as np
from sklearn.cross_validation import StratifiedKFold, LeaveOneOut
import DataHolder


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
    NR_SOURCES = 3
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
    ##build max capacity arrays for holding train/test spectra
    maxNr = 0
    for patient in dataset.keys():
        maxNr += dataset[patient]['Aligned'].shape[0]

    ##make a list of DataHolder instances for each fold
    folds = []
    for train_index, test_index in skf:
        ##1: split into training and testing set
        spectra_train,spectra_test = np.zeros((maxNr,195)),np.zeros((maxNr,195))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ##1.1.: Get training/testing spectra
        spectra_test_index = 0
        spectra_train_index = 0
        for index,patient in enumerate(dataset.keys()):
            if patient in X_train:
                spectra_train[spectra_train_index:spectra_train_index+dataset[patient]['Aligned'].shape[0],:] \
                    = dataset[patient]['Aligned']
                spectra_train_index+=dataset[patient]['Aligned'].shape[0]
            else:
                spectra_test[spectra_test_index:spectra_test_index+dataset[patient]['Aligned'].shape[0], :] \
                    = dataset[patient]['Aligned']
                spectra_test_index+=dataset[patient]['Aligned'].shape[0]
            spectra_train = spectra_train[:spectra_train_index,:]
            spectra_test = spectra_test[:spectra_test_index,:]
            folds.append(DataHolder(NR_SOURCES,dataset, X_train, X_test, y_train, y_test,
                                    spectra_train,spectra_test))
    ## process in parallel


            ##analyze data
