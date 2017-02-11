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
# Answer: same as [[h] for h in df['Height'].values] 
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
y_test_pred = regr.predict(x_test)
print ("Residual sum of squares: %.3f" %
       np.mean((y_test_pred - y_test) ** 2))

# Explained variance score: 1 is perfect prediction
print 'Variance score: %.3f' % regr.score(x_test, y_test)

# Plot data and best fit line
plt.scatter(x_test, y_test)
plt.plot(x_test, y_test_pred, color='red')
plt.title('Humans')
plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
plt.draw()

# Exercises:
#
# 1) load the data/housing-data.csv and try:
# - fit a linear regression on all numeric features
# - print coefficients
# - add text to the plot displaying the results
# - save the figure to a file

df = pd.read_csv('data/housing-data.csv')
features = ['sqft', 'bdrms', 'age']
X = df[features].values
y = df['price'].values

# split data in 2 parts (20% test / 80% train)
total_n_data = len(y)
ind = range(total_n_data)
np.random.shuffle(ind)
test_ind = ind[:total_n_data / 5]
train_ind = ind[total_n_data / 5:]

X_train = X[train_ind]
X_test = X[test_ind]
y_train = y[train_ind]
y_test = y[test_ind]

# create instance of linear regression class
regr = LinearRegression()

# fit linear regression
regr.fit(X_train, y_train)

# print to console
print "Fitted regression"

# The coefficients
print "Coefficients: %s" % regr.coef_
print "Intercept: %.2f" % regr.intercept_

# The mean square error
y_test_pred = regr.predict(X_test)
print ("Residual sum of squares: %.3f" %
       np.mean((y_test_pred - y_test) ** 2))

# Explained variance score: 1 is perfect prediction
print 'Variance score: %.3f' % regr.score(X_test, y_test)

# Visualize the best fit line. 
fig = plt.figure(figsize=(6, 6))
num_features = len(features)
b0 = regr.intercept_
b = [regr.coef_[k] for k in range(num_features)]
means = [df[f].mean() for f in features]
for i in range(num_features):
  ax = fig.add_subplot(3, 1, i + 1)
  ax.scatter(X_test[:, i], y_test)
  # To project onto one dimension, freeze the 2 other dims. Use the mean of 
  # their feature column to freeze them.
  if i == 0:
    projections = b0 + b[0] * X_test[:, 0] + b[1] * means[1] + b[2] * means[2]
  elif i == 1:
    projections = b0 + b[0] * means[0] + b[1] * X_test[:, 1] + b[2] * means[2]
  else:
    projections = b0 + b[0] * means[0] + b[1] * means[1] + b[2] * X_test[:, 2]
  ax.plot(X_test[:, i], projections, color='red')
  ax.set_title('Housing data')
  ax.set_xlabel(features[i])
  ax.set_ylabel('price')

plt.tight_layout()
plt.draw()
plt.show()


# 2) Check the code in the advanced folder:
#    02_regression.ipynb
