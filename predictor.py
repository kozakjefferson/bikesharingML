from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

import data as dt


class Predictor(object):

    data_folder = dt.Data("data")

    def __init__(self, dataset, predict):
        A = self.data_folder.preprocess(dataset)
        B = self.data_folder.preprocess(predict)
        self.data = A[:, :-1]
        self.predict = B[:-1]
        self.target = A[:, A.shape[1]-1]


        rfc = RandomForestRegressor()

        clf = rfc.fit(self.data, self.target)
        print clf.feature_importances_
        model = SelectFromModel(clf, threshold="0.2*mean", prefit=True)
        self.data = model.transform(self.data)
        print self.data

        k_fold = cross_validation.KFold(len(self.data), n_folds=10)

        scores = [rfc.fit(self.data[train], self.target[train]).score(
            self.data[test], self.target[test]) for train, test in k_fold]

        print scores


if __name__ == '__main__':
    predictor = Predictor("train.csv", "test.csv")
    #predictor.svm_predict("test.csv")
