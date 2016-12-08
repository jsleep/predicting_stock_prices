import csv, random
import numpy as np
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from dateutil.parser import parse

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
    'hidden_layer_sizes': (22, 11),
    'solver': 'lbfgs',
    'max_iter': 1000
}

dates = []
prices = []

def get_all_data():
    for comp_name in comp_names:
        with open('data/' + comp_name + '.csv') as csv_file:
            csvfileReader = csv.reader(csv_file)
            next(csvfileReader)
            for row in csvfileReader:
                if row[1] != '-':
                    data[comp_name][parse(row[0])] = float(row[1])

def get_features(comp, date):
    return [data[comp][date]]

def run_cross_fold_validation(comp, k_fold=10, data_subset=None):
    if data_subset:
        pass
    else:
        total = []

        for date in data[comp].keys():
            features = get_features(comp, date)
            if features:
                total.append((features, data[comp][date]))
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
            svr_lin = SVR(**svr_params)
            svr_lin.fit([x[0] for x in train], [x[1] for x in train])

            svr_avg_acc += svr_lin.score([x[0] for x in test], [x[1] for x in test])
            nnr_avg_r2 += nnr.score([x[0] for x in test], [x[1] for x in test])

        svr_avg_acc /= k_fold
        nnr_avg_r2 /= k_fold

        print('Avg SVM R^2 = ' + str(svr_avg_acc))
        print('Avg Neural Net R^2 = ' + str(nnr_avg_r2))


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
    run_cross_fold_validation('Oracle')


if __name__ == '__main__':
    main()