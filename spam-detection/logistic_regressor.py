#! /usr/bin/python3

import numpy as np
from math import log
from math import exp

TRAIN_FILEPATH = 'spambase/spam-train'      # file containing the training data
TEST_FILEPATH = 'spambase/spam-test'        # file containing the test data
STATS_FILEPATH = 'spambase/spambase.stats'  # file containing mean and stdev

class DataSet:
    def __init__(self, train, test, stats):
        """Create a new dataset object with training data, test data, and statistics."""

        self.train = np.mat(np.array(train).astype(np.float))
        self.test = np.mat(np.array(test).astype(np.float))
        self.stats = stats

    @classmethod
    def fromFilenames(cls, train_file, test_file, stats_file):
        """Construct a new DataSet object from training/test filenames."""

        def extractData(filename):
            """Extract the data from the given file and prepare it for regression."""

            def addIntercept(sample):
                """Add a `1` value for the regression intercept."""

                sample.insert(-1, 1)
                return sample

            with open(filename, 'r') as fp:
                return list(map(lambda l: addIntercept(l.split(',')), fp.readlines()))

        def loadStatistics(filename):
            """Return the mean and stdev information from `filename`."""

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

        return data[:, :-1]

    @staticmethod
    def labels(data):
        """Return the labels of the data in an np.mat."""

        return data[:, -1]

    @staticmethod
    def normalized(dset):
        """Normalizes the data according to mean and stdev in self.stats."""

        def helper(data, stats):
            """Normalizes a single dataset according to the stats argument."""

            newdata = np.mat(np.zeros(data.shape))
            for r in range(data.shape[0]):
                for c in range(data.shape[1] - 2):
                    newdata[r, c] = (data[r, c] - stats[c][0]) / stats[c][1]
                newdata[r, -2:] = data[r, -2:]

            return newdata

        return helper(dset.train, dset.stats), helper(dset.test, dset.stats)

    @staticmethod
    def transformed(dset):
        """Transforms the data according to x = log(x + 0.1)."""

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
        """Binarizes the data so that values > 0 are mapped to 1."""

        def helper(data):
            return np.mat(np.where(data > 0.0, 1, 0))

        return helper(dset.train), helper(dset.test)


class LogisticRegressor:
    def run(self, train, test):
        """Train and evaluate the model with the given data."""

        self.trainGradientDescent(DataSet.features(train), DataSet.labels(train))
        self.evaluate(train, test)

    def logLikelihood(self, X, y, w):
        """Computes the log of the likelihood for logistic regression."""

        result = 0
        for i in range(len(y)):
            if y[i] == 1:
                result += log(sigmoid(X[i] * w))
            else:
                result += log(1 - sigmoid(X[i] * w))

        return result

    def trainGradientDescent(self, X, y):
        """Optimize the model weights using gradient descent optimization."""

        steps = 0
        rate, eps = 5e-5, 7e-5
        w_new = np.mat(np.random.normal(0, 1, X.shape[1])).T
        vsigmoid = np.vectorize(sigmoid)

        print('Gradient descent with convergence threshold of {}...'.format(eps))
        while True:
            w_old = w_new
            w_new = w_old + rate * X.T * (y - vsigmoid(X * w_old))
            steps += 1

            if steps % 1000 == 0:
                print('  (Step: {}, Likelihood: {:08f}, dWeights: {:08f})'.format(
                    steps, self.logLikelihood(X, y, w_new), np.linalg.norm(w_new - w_old)))

            if np.linalg.norm(w_new - w_old) < eps:
                break
        print('Gradient descent completed in {} steps'.format(steps))

        self.w = w_new

    def classify(self, sample):
        """Given a sample, predict the sample's class using the model."""

        score = self.w.T * sample.T
        probability = sigmoid(score)
        
        if probability >= 0.5:
            return 1
        else:
            return 0

    def evaluateError(self, X, y):
        """Compute the error rate."""

        correct = 0
        for i in range(len(y)):
            if self.classify(X[i]) == y[i]:
                correct += 1

        return 1 - correct / len(y)

    def evaluate(self, train, test):
        """Evaluate the training and test errors and print them."""

        train_err = self.evaluateError(DataSet.features(train), DataSet.labels(train))
        test_err = self.evaluateError(DataSet.features(test), DataSet.labels(test))
        print('Evaluating Error...\n\
               \tTraining Error: {:.06f}\n\
               \tTest Error: {:.06f}\n'.format(train_err, test_err))

def sigmoid(r):
    """Maps r in (-Inf, Inf) to a probability in [0, 1]."""

    return 1 / (1 + exp(-r))

def main():
    dataset = DataSet.fromFilenames(TRAIN_FILEPATH, TEST_FILEPATH, STATS_FILEPATH)

    # Test the model with different types of data pre-processing
    ntrain, ntest = DataSet.normalized(dataset)
    ttrain, ttest = DataSet.transformed(dataset)
    btrain, btest = DataSet.binarized(dataset)

    lr = LogisticRegressor()

    print('Running Logistic Regressor on normalized data (mean = 0, var = 1)...')
    lr.run(ntrain, ntest)

    print('Running Logistic Regressor on transformed data (using log(X_ij + 0.1))...')
    lr.run(ttrain, ttest)

    print('Running Logistic Regressor on binarized data (using I(X_ij > 0))...')
    lr.run(btrain, btest)

if __name__ == '__main__':
    main()
