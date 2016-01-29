import sys

import randomForest as rfc


class Predictor(object):

    def __init__(self):
        self.concept = sys.argv[1]
        if len(sys.argv) > 2:
            self.p_flag = sys.argv[2]

    def svm(self):
        pass

    def rf(self):
        clf = rfc.RandomForest("train.csv")
        tree = clf.train()
        print tree
        if self.p_flag == "p":
            print clf.predict
            predictions = tree.predict(clf.predict)
            self.binning(predictions)

    def binning(self, predictions):
        _max = max(predictions)
        _min = min(predictions)
        for p in predictions:
            normalized = round((20-0)/(_max-_min)*(p-_max)+20)
            if normalized == 0:
                print "empty"
            elif 0 < normalized < 4:
                print "almost empty"
            elif 4 < normalized < 16:
                print "normal"
            elif 16 < normalized < 20:
                print "almost full"
            elif normalized == 20:
                print "full"

    def main(self):
        if self.concept == "svm":
            self.svm()
        elif self.concept == "rf":
            self.rf()


if __name__ == '__main__':
    predictor = Predictor()
    predictor.main()
