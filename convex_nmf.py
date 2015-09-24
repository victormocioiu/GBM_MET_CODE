__author__ = 'Sunny'
from sklearn.cluster import KMeans
import numpy as np
from numpy.linalg import inv


def sources(V, k=2, niter=1000):
    '''Compute sources using convex-NMF
    Parameters
    _________
    V: {array-like} shape = [n_features,n_samples],
    k: int,optional,default = 2
            Number of desired sources.
    niter: int,optional, default  = 1000
        Number of iterations until the algorithm stops

    Returns
    _________
    H_new: array, shape = [n_samples, n_sources]
        Mixing matrix
    W_new: array, shape = [n_samples, n_sources]
        Un-mixing matrix
    Sources: array, shape = [n_sources,n_features]
    '''

    Wini, Hini = _kmeans_init(V, k)
    return _convex_nmf(V, Wini, Hini, niter)


def _kmeans_init(V, k=2):
    ''' Initialize the mixing and un-mixing matrix using k-means
	Parameters
	_________
	V: {array-like} shape = [n_features,n_samples],
	k: int,optional,default = 2
			Number of desired sources.

	Returns
	_________

	W:{array-like} shape = [n_samples, n_sources]
		Pre-initialized un-mixing matrix
	H: {array-like} shape = [n_samples, n_sources]
		Pre-initialized mixing matrix
	'''

    N = V.shape[1]
    km = KMeans(n_clusters=k, init='random')
    IDX = km.fit_predict(V.T)
    H = np.zeros((N, k))
    for i in range(N):
        H[i, IDX[i]] = 1
    D = np.zeros((k, k))
    for i in range(k):
        D[i, i] = sum(H[:, i])
    D = inv(D)
    H = H + 0.2
    W = H.dot(D)
    return W, H


def _convex_nmf(V, W, H, niter=1000):
    '''
	Parameters
	_________
	V:{array-like} shape = [n_features,n_samples]
	W:{array-like} shape = [n_samples, n_sources]
		Pre-initialized un-mixing matrix
	H:{array-like} shape = [n_samples, n_sources]
		Pre-initialized mixing matrix
	niter: int,optional, default  = 1000
		Number of iterations until the algorithm stops

	Returns
	_________
	H_new: array, shape = [n_samples, n_sources]
		Mixing matrix
	W_new: array, shape = [n_samples, n_sources]
		Un-mixing matrix
	Sources: array, shape = [n_sources,n_features]
	'''
    Yp = 0.5 * (abs(V.T.dot(V)) + V.T.dot(V))
    Yn = 0.5 * (abs(V.T.dot(V)) - V.T.dot(V))
    Vold = V

    errs1 = np.zeros(niter)
    errs2 = np.zeros(niter)
    errr1 = np.zeros(niter)
    errr2 = np.zeros(niter)

    for t in range(niter):
        # update rules

        # update W(unmixing matrix)
        W = W * np.sqrt((Yp.dot(H) + Yn.dot(W.dot(H.T.dot(H)))) / (Yn.dot(H) + Yp.dot(W.dot(H.T.dot(H)))))

        # update H (mixing matrix)

        H = H * np.sqrt((Yp.dot(W) + H.dot(W.T.dot(Yn.dot(W)))) / (Yn.dot(W) + (H.dot(W.T.dot(Yp.T.dot(W))))))

        # compute errors
        Vr = V.dot(W.dot(H.T))  # reconstructed dataset after current iteration
        # squarred error between the original dataset and Vr
        errs1[t] = np.sum(np.power(V - Vr, 2))
        # sq err bet Vr and the reconstr dataset from prev iter
        errs2[t] = np.sum(np.power(Vold - Vr, 2))

        # relative error between the original dataset and Vr
        diff = np.sum(V - Vr)
        errr1[t] = float(np.mean(diff)) / np.mean(V)

        # relative error between the Vr and the previous
        diff = np.sum(Vold - Vr)
        errr2[t] = float(np.mean(diff)) / np.mean(Vold)

        Vold = Vr

        # check for convergence
        error = errs2[t]

        # stop_val = 1.e-5

        if np.abs(error) < 1.e-5:
            print 'Algorithm converged'
            return H, W, V.dot(W)  # mixing,un-mixing,sources

        if t == niter - 1:
            print 'Algorithm did not converge - do not fully trust the returned values'
            return H, W, V.dot(W)  # mixing,un-mixing,sources
