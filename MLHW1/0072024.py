import numpy as np
import pandas as pd


X = np.genfromtxt("hw01_data_points.csv", delimiter=",", dtype=str)
y = np.genfromtxt("hw01_class_labels.csv", delimiter=",", dtype=int)


# STEP 3
# first 50000 data points should be included to train
# remaining 44727 data points should be included to test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    # your implementation starts below
    # split the data points according to given instructions
    X_train = X[:50000, :]
    y_train = y[:50000]
    X_test = X[50000:, :]
    y_test = y[50000:]

    # your implementation ends above
    return (X_train, y_train, X_test, y_test)


X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# STEP 4
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below

    # prior probabilites equal to the number of elements in a given class divided by
    # the total number of classes

    classes = np.unique(y)
    K = np.size(np.unique(y))
    class_priors = np.zeros((K))

    for c in range(K):
        class_priors[c] = np.mean(y == classes[c])

    # your implementation ends above
    return (class_priors)


class_priors = estimate_prior_probabilities(y_train)
print(class_priors)


# STEP 5
# assuming that there are K classes and D features
# should return four numpy arrays with shape (K, D)
def estimate_nucleotide_probabilities(X, y):

    # your implementation starts below

    # get number of classes and number of features
    class_count = np.size(np.unique(y))
    feature_count = np.size(X[0, :])
    classes = np.unique(y)

    # create return values
    pAcd = np.empty((class_count, feature_count))
    pCcd = np.empty((class_count, feature_count))
    pGcd = np.empty((class_count, feature_count))
    pTcd = np.empty((class_count, feature_count))

    counts = np.empty((class_count))

    for i in range(len(classes)):
        counts[i] = np.count_nonzero(y == classes[i])

    for f in range(feature_count):
        for c in range(class_count):
            X2 = X[(y == classes[c])]
            pAcd[c, f] = (np.sum(X2[:, f] == 'A') / counts[c])
            pCcd[c, f] = (np.sum(X2[:, f] == 'C') / counts[c])
            pGcd[c, f] = (np.sum(X2[:, f] == 'G') / counts[c])
            pTcd[c, f] = (np.sum(X2[:, f] == 'T') / counts[c])

    # your implementation ends above
    return (pAcd, pCcd, pGcd, pTcd)


pAcd, pCcd, pGcd, pTcd = estimate_nucleotide_probabilities(X_train, y_train)
print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)


# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, pAcd, pCcd, pGcd, pTcd, class_priors):
    # your implementation starts below

    num_data = np.size(X[:, 0])
    num_feature = np.size(X[0, :])
    num_classes = len(class_priors)
    score_values = np.ones((num_data, num_classes))

    for c in range(num_classes):
        for d in range(num_data):
            for f in range(num_feature):
                if X[d][f] == 'A':
                    score_values[d][c] *= pAcd[c][f]
                elif X[d][f] == 'C':
                    score_values[d][c] *= pCcd[c][f]
                elif X[d][f] == 'G':
                    score_values[d][c] *= pGcd[c][f]
                elif X[d][f] == 'T':
                    score_values[d][c] *= pTcd[c][f]

            score_values[d][c] = np.log(score_values[d][c])
            score_values[d][c] += np.log(class_priors[c])

    # your implementation ends above
    return (score_values)


scores_train = calculate_score_values(
    X_train, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_train)

scores_test = calculate_score_values(
    X_test, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_test)


# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below

    classes = np.unique(y_truth)
    num_classes = len(classes)
    confusion_matrix = np.zeros((num_classes, num_classes), int)

    for i, j in zip(y_truth, scores):
        x1 = np.argmax(j)
        confusion_matrix[x1, i - 1] += 1

    # your implementation ends above
    return (confusion_matrix)


confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)
