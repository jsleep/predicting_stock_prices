import csv, random
import numpy as np
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor, MLPClassifier
import matplotlib.pyplot as plt
from dateutil.parser import parse
from datetime import datetime, timedelta

comp_names = [
    'Amazon', 'Facebook', 'IBM', 'Oracle', 'Twitter',
    'Apple', 'Google', 'MSFT'
]

data = {
    'Amazon': {},
    'Apple': {},
    'Facebook': {},
    'Google': {},
    'IBM': {},
    'MSFT': {},
    'Oracle': {},
    'Twitter': {}
}

svr_params = {
    'kernel': 'linear',
    'C': 1e3
}

nnr_params = {
    'hidden_layer_sizes': (8, 5),
    'solver': 'lbfgs',
    'activation': 'logistic',
    'max_iter': 10000
}

nnc_params = {
    'hidden_layer_sizes': (8, 5),
    'solver': 'lbfgs',
    'activation': 'logistic',
    'max_iter': 1000
}



def get_num_for_date(date):
    return (date - parse('Dec 5 2011')).days

def get_date_for_num(num):
    new_date = parse('Dec 5 2011') + timedelta(days=num)
    return new_date

def add_bus_days(datenum, num, comp):
    num_its = 0
    next_day = datenum + num
    while next_day not in data[comp].keys() and num_its < 10:
        next_day += 1
        num_its += 1
    return next_day

def minus_bus_days(datenum, num, comp):
    num_its = 0
    next_day = datenum - num
    while next_day not in data[comp].keys() and num_its < 10:
        next_day -= 1
        num_its += 1
    return next_day

def get_all_data():
    for comp_name in comp_names:
        with open('data/' + comp_name + '.csv') as csv_file:
            csvfileReader = csv.reader(csv_file)
            next(csvfileReader)
            for row in csvfileReader:
                if row[1] != '-':
                    data[comp_name][get_num_for_date(parse(row[0]))] = float(row[1])

DAYS_BACK = 10

def get_features(comp, date_num, diff):
    features = []
    features.append(date_num)
    for ndx in range(1, DAYS_BACK + 1):
        if minus_bus_days(date_num, ndx, comp) in data.keys():
            if diff:
                features.append(data[comp][minus_bus_days(date_num, ndx, comp)] -
                                data[comp][minus_bus_days(date_num, ndx + 1, comp)])
            else:
                features.append(data[comp][minus_bus_days(date_num, ndx, comp)])
        else:
            features.append(0)
    features.append(get_date_for_num(date_num).month)
    features.append(get_date_for_num(date_num).day + 30 * get_date_for_num(date_num).month)
    features.append(date_num)

    return features

def run_cross_fold_validation(comp, k_fold=10, data_subset=None, diff=False, plot=False, classifier_days_in_future=1):
    total = []
    total_class = []

    if data_subset is None:
        data_subset = sorted(data[comp].keys())
    if classifier_days_in_future > 1:
        data_subset = data_subset[:(-classifier_days_in_future + 1)]

    for date_num in data_subset:
        features = get_features(comp, date_num, diff)
        if features:
            total_class.append((features, (data[comp][date_num]) - (data[comp][minus_bus_days(date_num, 1, comp)]) > 0))
            if diff:
                total.append((features, (data[comp][add_bus_days(date_num, classifier_days_in_future - 1, comp)]) - (data[comp][minus_bus_days(date_num, 1, comp)])))
            else:
                total.append((features, data[comp][date_num]))
    random.shuffle(total)
    random.shuffle(total_class)

    svr_avg_acc = 0
    nnr_avg_r2 = 0

    nnc_avg_acc = 0

    best_nnr_acc = -1000
    best_nnc_acc = -1

    for ndx in range(k_fold):
        startndx = int(ndx / k_fold * len(total))
        endndx = int((ndx + 1) / k_fold * len(total))
        test = total[startndx:endndx]
        train = total[:startndx]
        train.extend(total[endndx:])

        test_class = total_class[startndx:endndx]
        train_class = total_class[:startndx]
        train_class.extend(total_class[endndx:])

        # Neural Net, because why not
        nnr = MLPRegressor(**nnr_params)
        nnr.fit([x[0] for x in train], [x[1] for x in train])

        nnc = MLPClassifier(**nnc_params)
        nnc.fit([x[0] for x in train_class], [x[1] for x in train_class])

        # Switch this to whatever you want, like below
        #svr_lin = SVR(**svr_params)
        #svr_lin.fit([x[0] for x in train], [x[1] for x in train])

        #svr_acc = svr_lin.score([x[0] for x in test], [x[1] for x in test])
        nnr_acc = nnr.score([x[0] for x in test], [x[1] for x in test])
        nnc_acc = nnc.score([x[0] for x in test_class], [x[1] for x in test_class])

        if nnr_acc > best_nnr_acc:
            best_nnr_acc = nnr_acc
            best_model = nnr

        if nnc_acc > best_nnc_acc:
            best_nnc_acc = nnc_acc
            best_class_model = nnc

        #svr_avg_acc += svr_acc
        nnr_avg_r2 += nnr_acc
        nnc_avg_acc += nnc_acc

        #print(nnr_acc, sep=", ")
        print(nnc_acc, sep=", ")

    #svr_avg_acc /= k_fold
    nnr_avg_r2 /= k_fold
    nnc_avg_acc /= k_fold

    #print('Avg SVM R^2 = ' + str(svr_avg_acc))
    print('Avg Neural Net R^2 = ' + str(nnr_avg_r2))
    print('Avg Neural Net Classifier Accuracy = ' + str(nnc_avg_acc))

    if plot:
        datenums = data_subset
        if diff:
            prices = [(data[comp][datenum] - data[comp][minus_bus_days(datenum, 1, comp)]) for datenum in datenums]
        else:
            prices = [data[comp][datenum] for datenum in datenums]

        plt.scatter(datenums, prices, color='black', label='Data', marker='o')  # plotting the initial datapoints
        if diff:
            plt.plot(datenums, best_model.predict([get_features(comp, datenum, diff) for datenum in datenums]),
                     color='red',label='NNR model')  # plotting the line made by the RBF kernel
        else:
            plt.plot(datenums, best_model.predict([get_features(comp, datenum, diff) for datenum in datenums]),
                     color='red', label='NNR model')  # plotting the line made by the RBF kernel
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Support Vector Regression')
        plt.legend()
        plt.show()


def predict_price(dates, prices, x):


    svr_lin = SVR(kernel= 'linear', C= 1e3)
    svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
    svr_rbf.fit(dates, prices) # fitting the data points in the models
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)

    plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints
    plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
    plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
    plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

def main():
    get_all_data()
    for name in data.keys():
        startdate = min([datenum for datenum in data[name].keys()])
        run_cross_fold_validation(name, data_subset=[datenum for datenum in data[name].keys() if datenum > DAYS_BACK + 1 + startdate],
                                  diff=True, plot=False, classifier_days_in_future=5)


if __name__ == '__main__':
    main()