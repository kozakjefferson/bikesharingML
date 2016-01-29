import sys

import randomForest as rfc
import svm


class Predictor(object):

    def __init__(self):
        self.concept = sys.argv[1]
        if len(sys.argv) > 2:
            self.p_flag = sys.argv[2]

    def svm(self):
        svr = svm.SVM("train.csv")
        clf = svr.train_with_features([0, 1])
        print clf
        if self.p_flag == "p":
            print svr.predict
            predictions = clf.predict(svr.predict)
            self.binning(svr.predict_original, predictions)

    def rf(self):
        clf = rfc.RandomForest("train.csv")
        tree = clf.train()
        print tree
        if self.p_flag == "p":
            print clf.predict
            predictions = tree.predict(clf.predict)
            self.binning(clf.predict_original, predictions)

    def binning(self, dates, predictions):
        _max = max(predictions)
        _min = min(predictions)
        for i, p in enumerate(predictions):
            day = dates[i][0]
            hour = dates[i][1]
            date = self.to_date(day, hour)
            normalized = round((20-0)/(_max-_min)*(p-_max)+20)
            if normalized == 0:
                print date + ": " + "empty"
            elif 0 < normalized < 4:
                print date + ": " + "almost empty"
            elif 4 < normalized < 16:
                print date + ": " + "normal"
            elif 16 < normalized < 20:
                print date + ": " + "almost full"
            elif normalized == 20:
                print date + ": " + "full"

    def main(self):
        if self.concept == "svm":
            self.svm()
        elif self.concept == "rf":
            self.rf()

    def to_date(self, day, hour):
        weekday = 'Monday   Tuesday  WednesdayThursday Friday   Saturday Sunday   '[(int(day)-1)*9:int(day)*9].strip()
        time = str(int(hour)) + ":00"
        date = weekday + " " + time
        return date


if __name__ == '__main__':
    predictor = Predictor()
    predictor.main()
