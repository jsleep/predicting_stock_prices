import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from dateutil.parser import parse

comp_names = [
    'Amazon', 'Facebook', 'IBM', 'Oracle', 'Twitter',
    'Apple', 'Google', 'MSFT', 'Samsung'
]

data = {
    'Amazon': {},
    'Apple': {},
    'Facebook': {},
    'Google': {},
    'IBM': {},
    'MSFT': {},
    'Oracle': {},
    'Samsung': {},
    'Twitter': {}
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


def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)	# skipping column names
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[1]))
    return

def predict_price(dates, prices, x):
    dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1

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

get_all_data()
print(data)