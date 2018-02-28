# Python 3

import numpy as np
import math

# mapping of class names to category labels
CLASS_MAP = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}

DATA_FILEPATH = 'iris-data.csv'

class DataSet:
    def __init__(self, training, test, dim):
        self.training = training
        self.test = test
        self.dim = dim

    @classmethod
    def loadFromFile(cls, filename):
        # read data from file
        with open(filename, 'r') as fp:
            raw_data = [cls.replaceClass(line.strip().split(','))
                        for line in fp.readlines()]

        # organize data as samples grouped by class/label
        split_data = [[], [], []]
        for sample in raw_data:
            split_data[sample[-1]].append(list(map(float, sample)))
        
        # split into training & test groups
        training, test = [], []
        for class_data in split_data:
            sep_index = int(0.8*len(class_data))
            training.append(class_data[:sep_index])
            test.append(class_data[sep_index:])

        dim = len(raw_data[0][:-1])  # number of features
        return cls(training, test, dim)

    @classmethod
    def newWithRemovedFeatures(cls, data, rm_indices):
        new_train = data.removeFeaturesFromData(data.training, rm_indices)
        new_test = data.removeFeaturesFromData(data.test, rm_indices)
        new_dim = data.dim - len(rm_indices)

        return cls(new_train, new_test, new_dim)

    @staticmethod
    def replaceClass(sample):
        sample[-1] = CLASS_MAP[sample[-1]]
        return sample

    def removeFeaturesFromData(self, data, remove_indices):
        def removeFeaturesFromSample(sample, remove_indices):
            filtered_sample = []
            for index, value in enumerate(sample):
                if index not in remove_indices:
                    filtered_sample.append(value)
            return filtered_sample

        filtered_data = [[], [], []]
        for class_label, class_data in enumerate(data):
            for sample in class_data:
                filtered_data[class_label].append(
                    removeFeaturesFromSample(sample, remove_indices)
                )
        return filtered_data


class GaussianDistribution:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.dim = len(mu)

    def probabilityDensity(self, x):
        return (
            math.exp(-0.5 * (x-self.mu).T * self.sigma.I * (x-self.mu))
            / math.sqrt((2 * math.pi)**self.dim * np.linalg.det(self.sigma))
        )


class QDAClassifier():
    def __init__(self, n_classes, features_independent=False):
        self.n_classes = n_classes
        self.features_independent = features_independent
        self.distributions = []
        self.prior_probability = [1/n_classes]*n_classes

    def run(self, data_set):
        self.train(data_set.training)
        printErrors(
            self.evaluateErrors(data_set.training),
            self.evaluateErrors(data_set.test)
        )

    def train(self, training_set):
        for class_data in training_set:
            data_vectors = [np.mat(x[:-1]).T for x in class_data]
            mu = sum(data_vectors) / len(data_vectors)

            if self.features_independent:
                sigma = np.mat(np.diag(np.squeeze(np.asarray(
                    sum([np.multiply((x-mu), (x-mu))  # element-wise mul
                         for x in data_vectors])
                    / len(data_vectors)
                ))))
            else:
                sigma = sum([(x-mu)*(x-mu).T  # outer product
                             for x in data_vectors]) / len(data_vectors)

            self.distributions.append(GaussianDistribution(mu, sigma))

    def classifySample(self, sample):
        best_prob, best_label = 0, -1
        sample_features = np.mat(sample).T[:-1]

        # find the distribution with highest probability to generate the sample
        for label, distribution in enumerate(self.distributions):
            prob = distribution.probabilityDensity(sample_features)
            if prob > best_prob:
                best_prob, best_label = prob, label

        return best_label

    def evaluateErrors(self, data_set):
        correct = total = 0
        for class_label, class_data in enumerate(data_set):
            for sample in class_data:
                predicted_label = self.classifySample(sample)
                if predicted_label == class_label:  # accumulate correct pred
                    correct += 1
                total += 1

        return 1 - (correct / total)  # error rate


class LDAClassifier(QDAClassifier):
    def train(self, training_set):
        super().train(training_set)

        # set sigma as the average of the trained sigmas
        lda_sigma = sum([d.sigma for d in self.distributions]) / self.n_classes
        for distribution in self.distributions:
           distribution.sigma = lda_sigma


def printErrors(train_error, test_error):
    print('\tTraining error: {:.02f}%'.format(100.0*train_error))
    print('\tTest error: {:.02f}%\n'.format(100.0*test_error))

def lookForFeatureRedundancy(n_classes, data_set):
    print('Testing the removal of one feature at a time:')
    for f_index in range(data_set.dim):
        print('Removing feature {}:'.format(f_index))
        rm_indices = [f_index]
        f_data_set = DataSet.newWithRemovedFeatures(data_set, rm_indices)
        lda = LDAClassifier(n_classes)
        lda.run(f_data_set)

    print('Testing the removal of all but one feature at a time:')
    for f_index in range(data_set.dim):
        print('Removing all except feature {}:'.format(f_index))
        rm_indices = list(range(data_set.dim))
        rm_indices.remove(f_index)  # remove all but f_index
        f_data_set = DataSet.newWithRemovedFeatures(data_set, rm_indices)
        lda = LDAClassifier(n_classes)
        lda.run(f_data_set)

def main():
    num_classes = 3

    print('Splitting the data into training (80%) and test (20%) sets...')
    data = DataSet.loadFromFile(DATA_FILEPATH)
    print('Done\n')

    print('LDA Classifier based on the training data')
    lda = LDAClassifier(num_classes)
    lda.run(data)

    print('QDA Classifier based on the training data')
    qda = QDAClassifier(num_classes)
    qda.run(data)

    print('\nExamining feature redundancy...')
    lookForFeatureRedundancy(num_classes, data)
    print()

    print('LDA Classifier assuming linearly independent features')
    lda_2 = LDAClassifier(num_classes, features_independent=True)
    lda_2.run(data)

    print('QDA Classifier assuming linearly independent features')
    qda_2 = QDAClassifier(num_classes, features_independent=True)
    qda_2.run(data)

if __name__ == '__main__':
    main()
