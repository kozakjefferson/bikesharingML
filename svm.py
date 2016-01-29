from sklearn import svm
from sklearn import cross_validation

import data as dt


class SVM(object):

    data_folder = dt.Data("data")

    def __init__(self, dataset):
        self.A = self.data_folder.preprocess(dataset)
        self.predict_original = self.data_folder.preprocess("test.csv")
        self.predict = self.data_folder.preprocess("test.csv")
        self.data = self.A[:, :-1]
        self.target = self.A[:, self.A.shape[1]-1]
        self.regressor = svm.SVR()

    def evaluate_with_feature(self, features):
        X = self.data_folder.truncate(self.A, features)

        print "Selected Features"
        print "-----------------"
        print
        print X
        print

        k_fold = cross_validation.KFold(len(X), n_folds=10)

        scores = [self.regressor.fit(X[train], self.target[train]).score(
            X[test], self.target[test]) for train, test in k_fold]

        print "10-fold cross validation scores"
        print "-------------------------------"
        print
        print scores
        print

    def train(self):
        self.regressor.fit(self.data, self.target)

        return self.regressor

    def train_with_features(self, features):
        X = self.data_folder.truncate(self.A, features)
        self.predict = self.data_folder.truncate(self.predict, features)

        self.regressor.fit(X, self.target)

        return self.regressor


if __name__ == '__main__':
    predictor = SVM("train.csv")
    predictor.evaluate_with_feature([0, 1])
