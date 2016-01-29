import csv
import datetime

import numpy as np


class Data(object):

    def __init__(self, folder):
        self.folder = folder

    def csv_str_to_float(self, line):
        float_line = []
        for i, item in enumerate(line[0:]):
            if i is 0:
                date = item.split(" ")[0].split("-")
                weekday = datetime.date(
                    int(date[0]), int(date[1]), int(date[2])).weekday()
                float_line.append(float(weekday))
                item = item[11:13]
            if i in [9, 10]:
                continue
            float_line.append(float(item))
        return float_line

    def csv_to_array(self, reader):
        for index, row in enumerate(reader):
            if index == 0:
                A = np.array(self.csv_str_to_float(row))
            else:
                newrow = self.csv_str_to_float(row)
                A = np.vstack([A, newrow])
        #A[:, A.shape[1]-1] = np.floor_divide(A[:, A.shape[1]-1], 50)
        return A

    def preprocess(self, data):
        with open(self.folder + '/' + data, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            reader.next()
            return self.csv_to_array(reader)

    def get_dates(self, test):
        with open(self.folder + '/' + test, 'rb') as csvfile:
            date_list = []
            reader = csv.reader(csvfile, delimiter=',')
            reader.next()
            for row in reader:
                date_list.append(row[0])
        return date_list

    def truncate(self, data, selection):
        for i, s in enumerate(selection):
            select = data[:, s]
            if i == 0:
                new_data = np.array(select).reshape(-1, 1)
            else:
                new_data = np.column_stack((new_data, select))
        return new_data


if __name__ == '__main__':
    data = Data("data")
    test = data.preprocess("train.csv")
    #data.truncate(test, [0, 3, 4, 5, 6, 7])
    #print data.get_dates("test.csv")
