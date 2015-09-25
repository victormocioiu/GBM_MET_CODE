__author__ = 'Sunny'
from skimage.measure import moments
import numpy as np
from collections import OrderedDict
from scipy.optimize import nnls
import numpy as np

from convex_nmf import sources
from scipy.stats import skew, kurtosis
from skimage.filters import threshold_otsu
from skimage.measure import regionprops


class DataHolder(object):
    '''Fancy data container with the added benefit of feature extraction
    DON'T FORGET TO WRITE THE DOCS STUPID

    '''
    def __init__(self, nr_sources,dataset, X_train, X_test, y_train, y_test, spectra_train,spectra_test):
        self.nr_sources = nr_sources
        self.dataset = dataset
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.spectra_train = spectra_train
        self.spectra_test = spectra_test
        ##
        self.sources = np.zeros((nr_sources,spectra_train.shape[1]))
        self.nnls_sources = np.zeros((nr_sources,spectra_train.shape[1]))
        self.train_features = np.zeros((spectra_train.shape[0],33))
        self.test_features = np.zeros((spectra_test.shape[0],33))
        self.train_grid = []
        self.test_grid = []

    def extract_features(self):
        GN,GN_hat = self.getSources()
        self.train_grids, self.test_grids = self.getGrids()
        self.train_features, self.test_features = self.getFeat(self.train_grids), self.getFeat(self.test_feats)


    def getSources(self):
        W, G, F = sources(self.spectra_train.T, k = self.nr_sources, niter=10000)
        W_hat = np.zeros((self.spectra_test.shape[0], self.nr_sources))
        for idx, spectra in enumerate(self.spectra_test):
            W_hat[idx, :] = nnls(F, spectra)[0]
        GN = W / W.sum(axis=1)[:, None]
        GN_hat = W_hat / W_hat.sum(axis=1)[:, None]
        self.sources = GN
        return GN, GN_hat

    def getGrids(self):
        supa_index_train = [0, 0, 0]
        supa_index_test = [0, 0, 0]
        train_grids = OrderedDict()
        test_grids = OrderedDict()
        ##for every patient compute some homogeneity score
        for patient in self.dataset:
            ##create a grid of max size, fill it with -1 for control purposes
            maps = [[], [], []]
            # for each dimension/source
            for i in range(self.nr_sources):
                ##get a grid
                grid = -1 * np.ones((max(self.dataset[patient]['Coordinates'])))
                grid = grid.squeeze()
                ## fill the grid with the corresponding G for those coordinates
                if patient in self.X_train:
                    for idx, coordinate in enumerate(reversed(self.dataset[patient]['Coordinates'])):
                        grid[coordinate[0] - 1, coordinate[1] - 1] = self.sources[supa_index_train[i] + idx, i]
                    supa_index_train[i] += len(self.dataset[patient]['Coordinates'])
                else:
                    for idx, coordinate in enumerate(reversed(self.dataset[patient]['Coordinates'])):
                        grid[coordinate[0] - 1, coordinate[1] - 1] = self.nnls_sources[supa_index_test[i] + idx, i]
                    supa_index_test[i] += len(self.dataset[patient]['Coordinates'])
                mini = min(self.dataset[patient]['Coordinates'])
                maxi = max(self.dataset[patient]['Coordinates'])
                grid = grid[mini[0] - 1:maxi[0], mini[1] - 1:maxi[1]]
                maps[i].append(grid)
            if patient in self.X_train:
                train_grids[patient] = np.rot90(np.dstack((maps[0][0], maps[1][0], maps[2][0])), 2)
            else:
                test_grids[patient] = np.rot90(np.dstack((maps[0][0], maps[1][0], maps[2][0])), 2)
            try:
                del maps
            except NameError:
                pass
        return train_grids, test_grids

    @staticmethod
    def getFeat(grids):
        feats = np.zeros((len(grids), 33 ))
        for idx, patient in enumerate(grids.iterkeys()):
            global_tresh = threshold_otsu(grids[patient])
            bin_glob = grids[patient] > global_tresh
            ##compute feats
            red_shape_descript = self.shape_feat(bin_glob[:, :, 0])
            red_shape_stat = self.grid_color_stat(grids[patient][:, :, 0])
            green_shape_descript = self.hape_feat(bin_glob[:, :, 1])
            green_shape_stat = self.grid_color_stat(grids[patient][:, :, 1])
            blue_shape_descript = self.shape_feat(bin_glob[:, :, 2])
            blue_shape_stat = self.grid_color_stat(grids[patient][:, :, 2])
            ##concatenate features
            patient_feats  = np.concatenate((red_shape_descript,red_shape_stat,
                                             green_shape_descript,green_shape_stat,
                                             blue_shape_descript,blue_shape_stat,))
            ##assign them to index
            feats[idx,:] = patient_feats
        return feats

    @staticmethod
    def shape_feat(bin_glob_1_color):
        shape_feats = np.zeros(7)
        try:
            shape_descripts = regionprops(bin_glob_1_color)[0]
            shape_feats[0] = shape_descripts.perimeter / shape_descripts.area
            shape_feats[1] = shape_descripts.eccentricity
            shape_feats[2] = shape_descripts.filled_area
            shape_feats[3] = shape_descripts.major_axis_length
            shape_feats[4] = shape_descripts.minor_axis_length
            shape_feats[5] = shape_descripts.solidity
            shape_feats[6] = shape_descripts.orientation
        except IndexError:
            pass
        return shape_feats

    @staticmethod
    def grid_color_stat(patient_grid_1_color):
        shape_stats = np.zeros(4)
        shape_stats[0] = np.mean(patient_grid_1_color.flatten())
        shape_stats[1] = np.std(patient_grid_1_color.flatten())
        shape_stats[2] = skew(patient_grid_1_color.flatten())
        shape_stats[3] = kurtosis(patient_grid_1_color.flatten())
        return shape_stats

