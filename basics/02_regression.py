import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# load height-weight dataset
df = pd.read_csv('data/weight-height.csv')

# create instance of linear regression class
regr = LinearRegression()

# what's the purpose of the next line?
# try to print out df['Height'].values and x
# to figure it out
x = df['Height'].values[:, np.newaxis]

y = df['Weight'].values

# split data in 2 parts (20% test / 80% train)
total_n_data = len(y)
ind = range(total_n_data)
np.random.shuffle(ind)
test_ind = ind[:total_n_data / 5]
train_ind = ind[total_n_data / 5:]

x_train = x[train_ind]
x_test = x[test_ind]
y_train = y[train_ind]
y_test = y[test_ind]

# fit linear regression
regr.fit(x_train, y_train)

# print to console
print "Fitted regression"
# The coefficients
print "Slope: %.2f" % regr.coef_
print "Intercept: %.2f" % regr.intercept_

# The mean square error
print "Residual sum of squares: %.3f" % \
      np.mean((regr.predict(x_test) - y_test) ** 2)

# Explained variance score: 1 is perfect prediction
print 'Variance score: %.3f' % regr.score(x_test, y_test)


# Plot data and best fit line
plt.scatter(x_test, y_test)
plt.plot(x_test, regr.predict(x_test), color='red')
plt.title('Humans')
plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
plt.draw()
plt.show()

# Exercises:
#
# 1) load the data/housing-data.csv and try:
# - fit a linear regression on all numeric features
# - print coefficients
# - add text to the plot displaying the results
# - save the figure to a file
#
# 2) Check the code in the advanced folder:
#    02_regression.ipynb
