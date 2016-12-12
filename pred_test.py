import csv, random
import numpy as np
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
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

def get_features(comp, date_num):
    features = []
    features.append(date_num)
    for ndx in range(1, 11):
        if minus_bus_days(date_num, ndx, comp) in data.keys():
            features.append(data[comp][minus_bus_days(date_num, ndx, comp)])
        else:
            features.append(0)
    features.append(get_date_for_num(date_num).month)
    features.append(get_date_for_num(date_num).day + 30 * get_date_for_num(date_num).month)

    return features

def run_cross_fold_validation(comp, k_fold=10, data_subset=None, diff=False):
    total = []

    if data_subset is None:
        data_subset = data[comp].keys()

    for date_num in data_subset:
        features = get_features(comp, date_num)
        if features:
            if diff:
                total.append((features, (data[comp][date_num]) - (data[comp][minus_bus_days(date_num, 1, comp)])))
            else:
                total.append((features, data[comp][date_num]))
    random.shuffle(total)

    svr_avg_acc = 0
    nnr_avg_r2 = 0

    for ndx in range(k_fold):
        startndx = int(ndx / k_fold * len(total))
        endndx = int((ndx + 1) / k_fold * len(total))
        test = total[startndx:endndx]
        train = total[:startndx]
        train.extend(total[endndx:])

        # Neural Net, because why not
        nnr = MLPRegressor(**nnr_params)
        nnr.fit([x[0] for x in train], [x[1] for x in train])

        # Switch this to whatever you want, like below
        #svr_lin = SVR(**svr_params)
        #svr_lin.fit([x[0] for x in train], [x[1] for x in train])

        #svr_acc = svr_lin.score([x[0] for x in test], [x[1] for x in test])
        nnr_acc = nnr.score([x[0] for x in test], [x[1] for x in test])

        #svr_avg_acc += svr_acc
        nnr_avg_r2 += nnr_acc

        print(nnr_acc, sep=", ")

    #svr_avg_acc /= k_fold
    nnr_avg_r2 /= k_fold

    #print('Avg SVM R^2 = ' + str(svr_avg_acc))
    print('Avg Neural Net R^2 = ' + str(nnr_avg_r2))

    datenums = data_subset
    if diff:
        prices = [(data[comp][datenum] - data[comp][minus_bus_days(datenum, 1, comp)]) for datenum in datenums]
    else:
        prices = [data[comp][datenum] for datenum in datenums]

    plt.scatter(datenums, prices, color='black', label='Data', marker='o')  # plotting the initial datapoints
    if diff:
        plt.plot(datenums, nnr.predict([get_features(comp, datenum) for datenum in datenums]),
                 color='red',label='NNR model')  # plotting the line made by the RBF kernel
    else:
        plt.plot(datenums, nnr.predict([get_features(comp, datenum) for datenum in datenums]),
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
    run_cross_fold_validation('Oracle', data_subset=[datenum for datenum in data['Oracle'].keys() if datenum > DAYS_BACK], diff=False)


if __name__ == '__main__':
    main()