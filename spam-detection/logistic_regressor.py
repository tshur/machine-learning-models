# Python 3

import numpy as np
from math import log
from math import exp

TRAIN_FILEPATH = 'spambase/spam-train'
TEST_FILEPATH = 'spambase/spam-test'
STATS_FILEPATH = 'spambase/spambase.stats'

class DataSet:
    def __init__(self, train, test, stats):
        self.train = np.mat(np.array(train).astype(np.float))
        self.test = np.mat(np.array(test).astype(np.float))
        self.stats = stats

    @classmethod
    def fromFilenames(cls, train_file, test_file, stats_file):
        """Construct a new DataSet object from training/test filenames."""

        def extractData(filename):
            def addIntercept(sample):
                sample.insert(-1, 1)
                return sample

            with open(filename, 'r') as fp:
                return list(map(lambda l: addIntercept(l.split(',')), fp.readlines()))

        def loadStatistics(filename):
            stats = []  # list of tuple(mean, stdev)
            with open(filename, 'r') as fp:
                next(fp)
                for line in fp:
                    if line:
                        stats.append(tuple(map(np.float64, line.strip().split())))

            return stats

        return cls(extractData(train_file), extractData(test_file), loadStatistics(stats_file))

    @staticmethod
    def features(data):
        """Return the features of the data in an np.mat."""

        return data[:,:-1]

    @staticmethod
    def labels(data):
        """Return the labels of the data in an np.mat."""

        return data[:,-1]

    @staticmethod
    def normalized(dset):
        def helper(data, stats):
            newdata = np.mat(np.zeros(data.shape))
            for r in range(data.shape[0]):
                for c in range(data.shape[1] - 2):
                    newdata[r, c] = (data[r, c] - stats[c][0]) / stats[c][1]
                newdata[r, -2:] = data[r, -2:]

            return newdata

        return helper(dset.train, dset.stats), helper(dset.test, dset.stats)

    @staticmethod
    def transformed(dset):
        def helper(data):
            newdata = np.mat(np.zeros(data.shape))
            for r in range(data.shape[0]):
                for c in range(data.shape[1] - 2):
                    newdata[r, c] = log(data[r, c] + 0.1)
                newdata[r, -2:] = data[r, -2:]

            return newdata

        return helper(dset.train), helper(dset.test)

    @staticmethod
    def binarized(dset):
        def helper(data):
            return np.mat(np.where(data > 0.0, 1.0, 0.0))

        return helper(dset.train), helper(dset.test)

def sigmoid(r):
    """Maps r in (-Inf, Inf) to a probability in [0, 1]"""

    return 1.0 / (1 + exp(-r))

class LogisticRegressor:
    def run(self, train, test):
        """Train and evaluate the model with the given data."""

        self.trainGradientDescent(DataSet.features(train), DataSet.labels(train))
        self.evaluate(train, test)

    def logLikelihood(self, X, y, w):
        sum_ = 0
        for i in range(len(y)):
            if y[i] == 1:
                sum_ += log(sigmoid(X[i] * w))
            else:
                sum_ += log(1 - sigmoid(X[i] * w))

        return sum_

    def trainGradientDescent(self, X, y):
        """Optimize the model weights using gradient descent optimization."""

        alpha, tol = 5e-5, 1e-4
        w_new = np.mat(np.random.normal(0, 1, X.shape[1])).T
        vsigmoid = np.vectorize(sigmoid)

        while True:
            w_old = w_new
            w_new = w_old + alpha * X.T * (y - vsigmoid(X * w_old))

            if np.linalg.norm(w_new - w_old) < tol:
                break

        self.w = w_new

    def classify(self, sample):
        """Given a sample, predict the sample's class using the model"""

        score = self.w.T * sample.T
        probability = sigmoid(score)
        
        if probability >= 0.5:
            return 1.0
        else:
            return 0.0

    def evaluateError(self, X, y):
        correct = 0
        for i in range(len(y)):
            if self.classify(X[i]) == y[i]:
                correct += 1

        print(correct, len(y))

        return 1 - correct / len(y)

    def evaluate(self, train, test):
        """Evaluate the training and test errors and print them."""

        train_err = self.evaluateError(DataSet.features(train), DataSet.labels(train))
        test_err = self.evaluateError(DataSet.features(test), DataSet.labels(test))
        print('\tTraining Error: {:.06f}\n\tTest Error: {:.06f}\n'.format(train_err, test_err))


def main():
    dataset = DataSet.fromFilenames(TRAIN_FILEPATH, TEST_FILEPATH, STATS_FILEPATH)
    # otrain, otest = dataset.train, dataset.test
    ntrain, ntest = DataSet.normalized(dataset)
    ttrain, ttest = DataSet.transformed(dataset)
    btrain, btest = DataSet.binarized(dataset)

    lr = LogisticRegressor()

    # print('Running Logistic Regressor on original, unaltered data...')
    # lr.run(otrain, otest)

    print('Running Logistic Regressor on normalized data (mean = 0, var = 1)...')
    lr.run(ntrain, ntest)

    print('Running Logistic Regressor on transformed data (using log(X_ij + 0.1))...')
    lr.run(ttrain, ttest)

    print('Running Logistic Regressor on binarized data (using I(X_ij > 0))...')
    lr.run(btrain, btest)

if __name__ == '__main__':
    main()
