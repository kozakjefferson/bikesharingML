from sklearn import svm
import data as dt


class Prediction(object):

    clf = svm.SVR()
    data = dt.Data("data")

    def __init__(self, train, test):
        A = self.data.preprocess(train)
        B = self.data.preprocess(test)
        self.X = self.data.truncate(A, [0, 3, 4, 5, 6, 7])
        self.y = A[:, A.shape[1]-1]
        self.test = self.data.truncate(B, [0, 3, 4, 5, 6, 7])

        #print self.X.shape, self.y.shape, self.test.shape

    def svm_train(self):
        self.clf.fit(self.X, self.y)

    def svm_predict(self, test):
        with open('result.csv', 'a') as f:
            f.write('datetime,count\n')
            dates = self.data.get_dates(test)
            for index, sample in enumerate(self.test):
                prediction = self.clf.predict(sample.reshape(1, -1))
                f.write(dates[index-1] + ', ' + str(prediction[0]) + '\n')


if __name__ == '__main__':
    predictor = Prediction("train.csv", "test.csv")
    predictor.svm_train()
    predictor.svm_predict("test.csv")
