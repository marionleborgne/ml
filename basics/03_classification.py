import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import np_utils
from keras.regularizers import l2
import matplotlib.pyplot as plt



def brute_force(X_train, y_train, X_test, y_test):
  # Constants
  hidden_layers_dim = 100
  dropout_ratio = 0.2
  input_dim = len(X_train[0])
  output_dim = np.max(y_train) + 1
  batch_size = 5
  num_epochs = 1000
  verbose = 1

  # One-hot encoding of the class labels.
  y_train = np_utils.to_categorical(y_train, output_dim)
  y_test = np_utils.to_categorical(y_test, output_dim)

  # Create model
  model = Sequential()
  model.add(Dense(hidden_layers_dim,
                  W_regularizer=l2(.01),
                  input_dim=input_dim, init='uniform', activation='relu'))
  model.add(Dropout(dropout_ratio))
  model.add(Dense(output_dim,
                  init='uniform', activation='softmax'))

  # For a multi-class classification problem
  model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop', metrics=['accuracy'])

  # Train
  model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=batch_size, nb_epoch=num_epochs, verbose=verbose)

  # Evaluate
  loss, accuracy = model.evaluate(X_test, y_test, verbose=verbose)
  print 'loss: %s' % loss
  print 'accuracy: %s' % accuracy
  print ''



# load the data
df = pd.read_csv('data/isp_data.csv')

# define features (X) and target (y)
X = df[['download', 'upload']]
y = df['label']

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,

                                                    random_state=0)
# Decision tree
dt_model = DecisionTreeClassifier(random_state=0)

# Random forest with grid search
param_grid = {
  'max_depth': np.arange(3, 5),
  'criterion': ["gini", "entropy"],
  'min_samples_split': np.arange(1, 5),
  'min_samples_leaf': np.arange(1, 5)
}
rf_model = GridSearchCV(RandomForestClassifier(random_state=0), param_grid)

# SVM
svm_model = svm.SVC(kernel='rbf')

models = [dt_model, rf_model, svm_model]
for model in models:
  # train the model
  model.fit(X_train, y_train)
  print "\nClassification Score: %0.3f" % model.score(X_test, y_test)

  # display the confusion matrix
  y_pred = model.predict(X_test)
  print "\n=======confusion matrix=========="
  print confusion_matrix(y_test, y_pred)

# With a deep network....
brute_force(X_train.values, y_train.values, X_test.values, y_test.values)


# # Exercises
#
# 1) try to improve the score changing the parameters
#    at the initialization of the DecisionTreeClassifier
#    things you can try:
#    - set the max_depth of the tree
#    - set the min_samples of the tree
#    - look in the documentation
# http://scikit-learn.org/stable/modules/generated/sklearn.tree
# .DecisionTreeClassifier.html
# class sklearn.tree.DecisionTreeClassifier(criterion='gini', splitter='best',
#       max_depth=None, min_samples_split=2, min_samples_leaf=1,
#       min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
#       max_leaf_nodes=None, class_weight=None)
#
# 2) Try changing classifier. Try for instance using any of:
#    - KNeighborsClassifier()
#    - SVC()
#    - LogisticRegression()
#
# 3) Check the code in the advanced folder:
#    03_classification.ipynb
