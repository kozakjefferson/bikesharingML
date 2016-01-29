import sys

import numpy as np

from sklearn import svm
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.cross_validation import KFold

import data as dt


class SVM(object):

    data_folder = dt.Data("data")

    def __init__(self, dataset, predict):
        A = self.data_folder.preprocess(dataset)
        B = self.data_folder.preprocess(predict)
        self.data = self.data_folder.truncate(A, [11, 12])
        self.predict = self.data_folder.truncate(B, [11, 12])
        self.target = A[:, A.shape[1]-1]

        # self.target
        # sys.exit(0)

        #print self.data.shape, self.target.shape

        # self.data = preprocessing.normalize(self.data, norm='l2')
        #self.target = preprocessing.normalize(self.target)

        # print self.data
        # print self.target

        svr = svm.SVR()
        self.clf = svr.fit(self.data, self.target)
        # k_fold = cross_validation.KFold(len(self.data), n_folds=10)

        # scores = [svr.fit(self.data[train], self.target[train]).score(
        #     self.data[test], self.target[test]) for train, test in k_fold]

        # print scores

    def svm_learn(self, test):
        pass

    def svm_predict(self, test):
        for index, sample in enumerate(self.predict):
            prediction = self.clf.predict(sample.reshape(1, -1))
            print prediction

        # test_norm = preprocessing.normalize(self.test)
        # with open('result.csv', 'a') as f:
        #     f.write('datetime,count\n')
        #     dates = self.data.get_dates(test)
        #     for index, sample in enumerate(test_norm):
        #         prediction = self.clf.predict(sample.reshape(1, -1))
        #         f.write(dates[index-1] + ', ' + str(prediction[0]) + '\n')


if __name__ == '__main__':
    predictor = SVM("train.csv", "test.csv")
    predictor.svm_predict("test.csv")
