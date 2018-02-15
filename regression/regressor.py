# Python 3

import numpy as np
from math import sqrt

TRAIN_FILEPATH = 'crime-train.txt'
TEST_FILEPATH = 'crime-test.txt'

class DataSet:
    def __init__(self, train, test):
        self.train = np.mat(np.array(train).astype(np.float))
        self.test = np.mat(np.array(test).astype(np.float))

    @classmethod
    def fromFilenames(cls, train_file, test_file):
        """Construct a new DataSet object from training/test filenames."""

        def extractData(filename):
            with open(filename, 'r') as fp:
                next(fp)
                return list(map(lambda l: l.split() + [1], fp.readlines()))

        return cls(extractData(train_file), extractData(test_file))

    @staticmethod
    def folds(data, k):
        """Split the given data into k folds."""

        return np.array_split(data, k)

    @staticmethod
    def features(data):
        """Return the features of the data in an np.mat."""

        return data[:,1:]

    @staticmethod
    def labels(data):
        """Return the labels of the data in an np.mat."""

        return data[:,0]


class RidgeRegressor:
    """This class implements Ridge Regression.

    The hyper-parameter, lambda, is optimized using k-fold cross-validation. The
    weights of the model itself can be trained using an analytic approach in
    `trainClosedForm()` or an iterative approach in `trainGradientDescent`.
    """

    def run(self, train, test, train_func):
        """Run CV, train, and evaluate the model."""

        self.kfoldCrossValidation(k=5, train=train, train_func=train_func)
        train_func(DataSet.features(train), DataSet.labels(train))
        self.evaluate(train, test)

    def trainClosedForm(self, X, y):
        """Find the optimal weights analytically using the closed form loss optimization."""

        self.w = (X.T * X + self.lam * np.mat(np.eye(X.shape[1]))).I * X.T * y

    def trainGradientDescent(self, X, y):
        """Optimize the model weights using gradient descent optimization."""

        alpha, tol = 5e-5, 1e-6
        w_new = np.mat(np.random.normal(0, 1, X.shape[1])).T

        while True:
            w_old = w_new
            w_new = w_old + alpha * (X.T * (y - X * w_old) - self.lam * w_old)

            if np.linalg.norm(w_new - w_old) < tol:
                break
        self.w = w_new

    def kfoldCrossValidation(self, k, train, train_func):
        """Estimate hyper-parameter lambda using leave-one-out CV with k-folds (LOOCV)."""

        folds = DataSet.folds(train, k)  # array of k folds
        best = None
        lam = 400
        for _ in range(10):
            self.lam = lam
            error = 0

            # Cross-Validate by removing one fold at a time for validation
            for i in range(k):
                train, valid = np.vstack(folds[:i] + folds[i+1:]), folds[i]
                train_func(DataSet.features(train), DataSet.labels(train))
                error += self.evaluateSSE(DataSet.features(valid), DataSet.labels(valid))

            # Find best lambda by finding the lowest sum of MSE (faster than sum(RMSE/k))
            if not best or error < best[1]:
                best = (lam, error)
            lam /= 2
        self.lam = best[0]

    def predict(self, sample):
        """Given a sample, predict the output using the model"""

        return self.w.T * sample.T

    def evaluateSSE(self, X, y):
        """Evaluate the Sum of Squared Errors."""

        error = 0
        for i, val in enumerate(y):
            error += (self.predict(X[i]) - val)**2
        return np.asscalar(error)

    def evaluateRMSE(self, X, y):
        """Evaluate the Root Mean Squared Error."""

        return sqrt(self.evaluateSSE(X, y) / len(y))

    def evaluate(self, train, test):
        """Evaluate the training and test RMSE and print them."""

        if self.lam > 0:
            print('\tLambda: {}'.format(self.lam))
        train_err = self.evaluateRMSE(DataSet.features(train), DataSet.labels(train))
        test_err = self.evaluateRMSE(DataSet.features(test), DataSet.labels(test))
        print('\tTraining RMSE: {:.06f}\n\tTest RMSE: {:.06f}\n'.format(train_err, test_err))


class LinearRegressor(RidgeRegressor):
    """This class implements Linear Regression as a special case of Ridge Regression.

    The `LinearRegressor` class derives all methods except `run()` from the
    `RidgeRegressor` since the objective function is the same if lambda is 0. The
    weights of the model itself can be trained using an analytic approach in
    `trainClosedForm()` or an iterative approach in `trainGradientDescent`.
    """

    def run(self, train, test, train_func):
        """Train the model and evaluate."""

        self.lam = 0
        train_func(DataSet.features(train), DataSet.labels(train))
        self.evaluate(train, test)


def main():
    data_set = DataSet.fromFilenames(TRAIN_FILEPATH, TEST_FILEPATH)
    train, test = data_set.train, data_set.test
    lr, rr = LinearRegressor(), RidgeRegressor()

    # print('Linear Regression with closed form optimization:')
    # lr.run(train, test, lr.trainClosedForm)

    print('Ridge Regression with k-fold cross-validation and closed form optimization:')
    rr.run(train, test, rr.trainClosedForm)
    rr.checkWeights()

    # print('Linear Regression with gradient descent optimization:')
    # lr.run(train, test, lr.trainGradientDescent)

    # print('Ridge Regression with k-fold cross-validation and gradient descent optimization:')
    # rr.run(train, test, rr.trainGradientDescent)

if __name__ == '__main__':
    main()
