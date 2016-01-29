from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

import data as dt


class RandomForest(object):

    data_folder = dt.Data("data")

    def __init__(self, dataset):
        self.A = self.data_folder.preprocess(dataset)
        self.B = self.data_folder.preprocess(predict)
        self.data = self.A[:, :-1]
        self.target = self.A[:, self.A.shape[1]-1]

    # automatic feature selection
    def evaluate(self):
        rfc = RandomForestRegressor()

        clf = rfc.fit(self.data, self.target)
        model = SelectFromModel(clf, threshold="0.2*mean", prefit=True)
        self.data = model.transform(self.data)

        print "Selected Features"
        print "-----------------"
        print
        print self.data
        print

        k_fold = cross_validation.KFold(len(self.data), n_folds=10)

        scores = [rfc.fit(self.data[train], self.target[train]).score(
            self.data[test], self.target[test]) for train, test in k_fold]

        print "10-fold cross validation scores"
        print "-------------------------------"
        print
        print scores
        print

    # Evaluate tree with selected feature
    def evaluate_with_feature(self, features):
        X = self.data_folder.truncate(self.A, features)

        print "Selected Features"
        print "-----------------"
        print
        print X
        print

        k_fold = cross_validation.KFold(len(X), n_folds=10)

        rfc = RandomForestRegressor()

        scores = [rfc.fit(X[train], self.target[train]).score(
            X[test], self.target[test]) for train, test in k_fold]

        print "10-fold cross validation scores"
        print "-------------------------------"
        print
        print scores
        print

    def train(self, features):
        X = self.data_folder.truncate(self.A, features)

        rfc = RandomForestRegressor()
        rfc.fit(X, self.target)

        return rfc


if __name__ == '__main__':
    predictor = RandomForest("train.csv")
    predictor.evaluate()
