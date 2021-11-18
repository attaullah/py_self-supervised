import numpy as np
from time import time
from scipy.spatial.distance import cdist, pdist, squareform
from scipy import stats
# import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import zoom
from scipy import ndimage
# from Visualizations import plot_clusters,  visualize, plot_confusion_matrix, verify_image_labels
# from tensorflow.keras.callbacks import Callback


def estimating_sigma(dm):
    # x,mu,sigma = featureNormalize(dm)
    x = dm
    m = x.shape[0]
    n = np.floor(0.5 * m)
    index = np.ceil(np.random.rand(1, int(n)) * m).T
    index2 = np.ceil(np.random.rand(1, int(n)) * m).T
    index = index.astype('int').flatten()
    index2 = index2.astype('int').flatten()
    index -= 1   # match matlab indexing if last index selected
    index2 -= 1   # match matlab indexing if last index selected
    temp = x[index, :] - x[index2, :]
    dist = np.sum((temp * temp).T, axis=0).T
    dist = dist[dist != 0]
    gamma = np.quantile(dist, [0.9, 0.5, 0.1], interpolation='linear')  # np.power ( , -1)
    
    return gamma


def get_linear_model(name='knn', n=2, solver='liblinear', max_iter=200):
    if name == 'rf':
        return RandomForestClassifier(random_state=0)
    elif name == 'lda':
        return LinearDiscriminantAnalysis()
    elif name == 'lr':
        return LogisticRegression(solver=solver, max_iter=max_iter)
    elif name == 'qda':
        return QuadraticDiscriminantAnalysis()
    else:
        return KNeighborsClassifier(n_neighbors=n)


def linear_test_accuracy(labeled_train_feat, train_labels, test_image_feat, test_labels, name='knn', fn=False, n=2, verbose=False):
    true_test_labels = np.array(test_labels)
    t0 = time()
    if fn:
        labeled_train_feat, _, _ = featureNormalize(labeled_train_feat)
        test_image_feat, _, _ = featureNormalize(test_image_feat)
    clf = get_linear_model(name, n)
    clf.fit(labeled_train_feat, train_labels)
    assigned_test_labels = clf.predict(test_image_feat)
    accuracy = accuracy_score(true_test_labels, assigned_test_labels) * 100.
    if verbose:
        print('Accuracy : {:.2f}% '.format(accuracy))
        print("Time taken by error computation  {:.2f} minutes".format((time() - t0) / 60))
    return assigned_test_labels, np.round(accuracy, 2)


def llgc_w(x, sigma):
    pd = pdist(x, 'sqeuclidean')
    denom = 2. * np.square(sigma)
    pd = pd.astype('float32')
    pd /= -denom
    return squareform(np.exp(pd))


def llgc_w_simplified(x, sigma):
    l = pdist(x, 'sqeuclidean')
    l = l.astype('float32')
    # print(' size of distance matrix', sys.getsizeof(l))
    return squareform(np.exp(-sigma*l))


def calculate_S(W):
    d = np.sum(W, axis=1)
    D = np.sqrt(d * d[:, np.newaxis])
    return np.divide(W, D, where=D != 0)


def llgc_meta(dm1, dm0, Y_input, Y_original, init=0, lbls_all=False, n_iter=10, alpha=0.99, sigma=.8,
              verbose=False,  n_classes=10, cm=False):
      
    size_of_labeled = len(dm0)  
    dm = np.concatenate([dm0, dm1])
    
    n = len(Y_original)
    # print('****** LLGC shape ', dm.shape, ' Y_input  ', Y_input.shape, ' labeled = ', size_of_labeled)
    
    # labels
    if lbls_all:
        Y = np.eye(n_classes)[Y_input]  # input_data.dense_to_one_hot(Y_input, n_classes)
    else:
        Y_input_lab = np.eye(n_classes)[Y_input]  # input_data.dense_to_one_hot(Y_input, n_classes)
        Y_input_unlab = np.zeros((len(dm) - len(Y_input_lab), Y_input_lab.shape[1])) +init
        Y = np.concatenate([Y_input_lab, Y_input_unlab], 0)
    
    # step -1 Initializations: construct affinity matrix W
    sigma_new = estimating_sigma(dm)
    print('estimated sigma values :  ', sigma_new)
    # sigma = sigma_new[-1]
    W = llgc_w_simplified(dm, sigma)
    # W = llgc_w(dm, sigma)
    W = W.astype('float32')
    dims = dm.shape[1]
    del dm, dm0, dm1

    # step -2 Calculate S = D^-1/2 * W * D^-1/2
    t0 = time()
    S = calculate_S(W)
    del W
    # Step -3 Iteration 0 F(t+1) = S.F(t)*alpha + (1-alpha)*Y
    t0 = time()
    F = np.dot(S, Y) * alpha + (1 - alpha) * Y  # For iter==0: F(0)=Y
    # Step -4 calculate labels for unlabeled y^=argmax F
    res = np.argmax(F, 1)
    prob = np.max(F, 1)
    count_t = np.count_nonzero(res[size_of_labeled:] == Y_original[size_of_labeled:])
    acc = float(count_t) / (n - size_of_labeled)
    template = ' ****** LLGC Initial accuracy {:.4f} for labeled {} ,alpha = {} sigma = {}, dimensions = {}'
    print(template.format(acc, size_of_labeled, alpha, sigma, dims))

    if cm:
        cm = confusion_matrix(Y_original[size_of_labeled:], res[size_of_labeled:])
        plot_confusion_matrix(cm, classes=range(10), title='Confusion matrix LLGC it=1', verbose=verbose)
        plt.show() 
    for t in tnrange(n_iter):
        F = np.dot(S, F) * alpha + (1 - alpha) * Y
        if t < n_iter:
            res = np.argmax(F, 1)
            count_t = np.count_nonzero(res[size_of_labeled:] == Y_original[size_of_labeled:])
            acc = float(count_t) / (n - size_of_labeled)
            if verbose:
                print('after {} iterations, accuracy {:.4f}'.format(t+1, acc))
            # logger.info('after iterations {}, accuracy {:.4f}'.format(t, acc))

    res = np.argmax(F, 1)
    prob = np.max(F, 1)
    count_t = np.count_nonzero(res[size_of_labeled:] == Y_original[size_of_labeled:])
    acc = float(count_t) / (n - size_of_labeled)
    print('After {} iterations accuracy {:.4f} on {} labeled'.format(n_iter, acc, size_of_labeled))

    if cm:
        cm = confusion_matrix(Y_original[size_of_labeled:], res[size_of_labeled:])
        plot_confusion_matrix(cm, classes=range(10), title='Confusion matrix LLGC it=N', verbose=verbose)
        plt.show() 

    return res, prob, acc


# @jit(parallel=True, fastmath=True)
def llgc_meta_numba(dm1, dm0, Y_input, Y_original, init=0, lbls_all=False, alpha=0.99, sigma=.8, verbose=False,
                    n_classes=10, n_iter=10):

    size_of_labeled = len(dm0)
    X = np.concatenate([dm0, dm1])

    n = len(Y_original)
    # labels
    if lbls_all:
        Y = np.eye(n_classes)[Y_input]  # input_data.dense_to_one_hot(Y_input, n_classes)
    else:
        Y_input_lab = np.eye(n_classes)[Y_input]  # input_data.dense_to_one_hot(Y_input, n_classes)
        Y_input_unlab = np.zeros((len(X) - len(Y_input_lab), Y_input_lab.shape[1])) + init
        Y = np.concatenate([Y_input_lab, Y_input_unlab], 0)

    print('****** LLGC_N shape ', X.shape, ' Y_input  ', Y_input.shape, ' labeled = ', size_of_labeled)

    # step -1 Initializations: construct affinity matrix W
    sigma_new = estimating_sigma(X)
    print('estimated sigma values :  ', sigma_new)
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float32)
    for i in nb.prange(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = d  # np.sqrt(d)
    l = D
    W = np.exp(-sigma * l)  # squareform
    # step -2 Calclulate S = D^-1/2 * W * D^-1/2
    d = np.sum(W, axis=1)
    D = np.sqrt(d * d[:, np.newaxis])
    S = np.divide(W, D, where=D != 0)
    # Step -3 Iteration 0 F(t+1) = S.F(t)*alpha + (1-alpha)*Y
    F = np.dot(S, Y) * alpha + (1 - alpha) * Y  # For iter==0: F(0)=Y
    # Step -4 calculate labels for unlabeled y^=argmax F
    res = np.argmax(F, 1)
    prob = np.max(F, 1)

    count_t = np.count_nonzero(res[size_of_labeled:] == Y_original[size_of_labeled:])
    acc = float(count_t) / (n - size_of_labeled)

    print('Initial accuracy {:.4f} for labeled {} ,alpha = {} sigma = {}'.format(acc, size_of_labeled
                                                                                 , alpha, sigma))
    # print('Initial score min {:.4f} max {:.4f} '.format(np.min(prob), np.max(prob)))
    cm = confusion_matrix(y_true=Y_original[size_of_labeled:], y_pred=res[size_of_labeled:])
    plot_confusion_matrix(cm, classes=range(10), title='Confusion matrix LLGC it=1', verbose=verbose)
    if not verbose:
        plt.show()
    for t in tnrange(n_iter):
        F = np.dot(S, F) * alpha + (1 - alpha) * Y
        if t < n_iter:
            res = np.argmax(F, 1)
            count_t = np.count_nonzero(res[size_of_labeled:] == Y_original[size_of_labeled:])
            acc = float(count_t) / (n - size_of_labeled)
            if verbose:
                print('after {} iterations, accuracy {:.4f}'.format(t + 1, acc))
            # logger.info('after iterations {}, accuracy {:.4f}'.format(t, acc))

    res = np.argmax(F, 1)
    prob = np.max(F, 1)
    count_t = np.count_nonzero(res[size_of_labeled:] == Y_original[size_of_labeled:])
    acc = float(count_t) / (n - size_of_labeled)
    print('After {} iterations accuracy {:.4f} on {} labeled'.format(n_iter, acc, size_of_labeled))
    cm = confusion_matrix(y_true=Y_original[size_of_labeled:], y_pred=res[size_of_labeled:])
    plot_confusion_matrix(cm, classes=range(10), title='Confusion matrix LLGC it=N', verbose=verbose)
    if not verbose:
        plt.show()
        # clean up
    # del F, S

    return res, prob, acc


def label_spreading(dm1, dm0, Y_input, Y_original, lbls_all=False, n_iter=15, alpha=0.99, sigma=1.2, verbose=False,
                    spreading=True, kernel='rbf', neighbors=10, cm=False):
      
    size_of_labeled = len(dm0)  
    dm = np.concatenate([dm0, dm1])
    n = len(Y_original)
    del dm0, dm1  # cleanup
    # ##  labels
    if lbls_all:
        Y = Y_input
    else:
        Y_input_lab = Y_input
        Y_input_unlab = np.full((len(dm) - len(Y_input_lab)), -1)
        Y = np.concatenate([Y_input_lab, Y_input_unlab], 0)
    del Y_input    # cleanup
    sigma_new = estimating_sigma(dm)
    print('estimated sigma values :  ', sigma_new)
    if spreading:
        template = '****** Label Spreading shape = {}, Y_input shape = {} labeled = {} alpha = {} sigma = {}'
        print(template.format(dm.shape,  Y.shape, size_of_labeled, alpha, sigma))
        lp_model = LabelSpreading(kernel=kernel, n_neighbors=neighbors, alpha=alpha, gamma=sigma,
                                                    max_iter=n_iter)
    else:
        template = '****** Label Propagation shape = {}, Y_input shape = {} labeled = {} alpha = {} sigma = {}'
        print(template.format(dm.shape, Y.shape, size_of_labeled, alpha, sigma))
        lp_model = LabelPropagation(kernel=kernel, n_neighbors=neighbors, gamma=sigma, max_iter=n_iter
                                                      )
    lp_model.fit(dm, Y)
    predicted_labels = lp_model.transduction_  # [size_of_labeled:]
    # true_labels = Y_original[size_of_labeled:]
    acc = accuracy_score(Y_original[size_of_labeled:], predicted_labels[size_of_labeled:])
    print('Accuracy  {:.4f} after n_iter {}'.format(acc, n_iter))

    if verbose:
        print(classification_report(Y_original[size_of_labeled:], predicted_labels[size_of_labeled:]))

    if cm:
        print("Confusion matrix")
        cm1 = confusion_matrix(Y_original[size_of_labeled:], predicted_labels[size_of_labeled:], labels=lp_model.classes_)
        # print(cm1)
        plot_confusion_matrix(cm1, classes=range(10), title='Confusion matrix Label spreading it=N', verbose=verbose)
    F = lp_model.label_distributions_  # lp_model.predict_proba(dm)
    prob = np.max(F, 1)
    # try:
    #     np.testing.assert_array_equal(res[size_of_labeled:], predicted_labels)
    # except:
    #     print('Assertion!!!! arrays are not the same...!')
    # del dm0, dm1
    return predicted_labels, prob, acc
    

def featureNormalize(X, epsilon=1e-8):
    X_norm = np.copy(X)
    mu = np.zeros((X.shape[1]))
    sigma = np.zeros((X.shape[1]))
    for i in range(X.shape[1]):
        mu[i] = np.mean(X[:, i])
        sigma[i] = np.std(X[:, i], ddof=1)
        # print (sigma)
        if sigma[i] == 0:
            sigma[i] = epsilon
        X_norm[:, i] = (X[:, i] - float(mu[i]))/float(sigma[i])
    return X_norm, mu, sigma


def global_contrast_normalize(images, scale=55, eps=1e-10):
    images = images.astype('float32')
    n, h, w, c = images.shape
    # Flatten images to shape=(nb_images, nb_features)
    images = images.reshape((n, h*w*c))
    # Subtract out the mean of each image
    images -= images.mean(axis=1, keepdims=True)
    # Divide out the norm of each image
    per_image_norm = np.linalg.norm(images, axis=1, keepdims=True)
    # Avoid divide-by-zero
    per_image_norm[per_image_norm < eps] = 1.0
    return float(scale) * images / per_image_norm


def zca_whitener(images, identity_scale=0.1, eps=1e-10):
    """Args:
        images: array of flattened images, shape=(n_images, n_features)
        identity_scale: scalar multiplier for identity in SVD
        eps: small constant to avoid divide-by-zero
    Returns:
        A function which applies ZCA to an array of flattened images
    """
    image_covariance = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(
        image_covariance + identity_scale * np.eye(*image_covariance.shape)
    )
    zca_decomp = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + eps)), U.T))
    image_mean = images.mean(axis=0)
    return lambda x: np.dot(x - image_mean, zca_decomp)

def date_diff_in_seconds(dt2, dt1):
  timedelta = dt2 - dt1
  return timedelta.days * 24 * 3600 + timedelta.seconds

def dhms_from_seconds(seconds):

    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return days, hours, minutes, seconds

def program_duration(dt1, prefix=''):
    dt2= datetime.now()
    dtwithoutseconds = dt2.replace(second=0, microsecond=0)
    seconds = date_diff_in_seconds(dt2, dt1)
    abc = dhms_from_seconds(seconds)
    if abc[0] > 0:
        text = " {} days, {} hours, {} minutes, {} seconds".format(abc[0], abc[1], abc[2], abc[3])
    elif abc[1] > 0:
        text = " {} hours, {} minutes, {} seconds".format(abc[1], abc[2], abc[3])
    elif abc[2] > 0:
        text = "  {} minutes, {} seconds".format(abc[2], abc[3])
    else:
        text = "  {} seconds".format(abc[2], abc[3])
    return prefix + text + ' at ' + str(dtwithoutseconds)
