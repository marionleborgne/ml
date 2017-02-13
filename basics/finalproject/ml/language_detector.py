"""Build a language detector model

The goal of this exercise is to train a linear classifier on text features
that represent sequences of up to N consecutive characters so as to be
recognize natural languages by using the frequencies of short character
sequences as 'fingerprints'.

The script saves the trained model to disk for later use
"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD
# Adapted by: Francesco Mosconi
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score)
from sklearn.model_selection import (ShuffleSplit, cross_val_score, 
                                     learning_curve)
from sklearn.preprocessing import StandardScaler

def plot_learning_curve(model, features, label):
    tsz = np.linspace(0.1, 1, 10)
    train_sizes, train_scores, test_scores = learning_curve(model, features, label,
                                                            train_sizes=tsz)
    
    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis=1), 'ro-', label="Train Scores")
    plt.plot(train_sizes, test_scores.mean(axis=1), 'go-', label="Test Scores")
    plt.title('Learning Curve: Logistic Regression')
    plt.ylim((0.5, 1.0))
    plt.legend()
    plt.draw()
    
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# The training data folder must be passed as first argument
try:
    dataset = load_files('./wikidata/short_paragraphs')
except OSError as ex:
    print ex
    print "Couldn't import the data, try running `python fetch_data.py` first "
    exit(-1)

# TASK: Split the dataset in training and test set
# (use 20% of the data for test):

features = dataset['data']
label = dataset['target']
feature_names = dataset['target_names']

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    label,
                                                    test_size=.2,
                                                    random_state=0)

# TASK: Build a an vectorizer that splits strings into sequence of 1 to 3
# characters instead of word tokens using the class TfidfVectorizer
transformer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), 
                              use_idf=True)

# TASK: Use the function make_pipeline to build a
#       vectorizer / classifier pipeline using the previous analyzer
#       and a classifier of choice.
#       The pipeline instance should be stored in a variable named model
clf = MLPClassifier(hidden_layer_sizes=20)
scaler = StandardScaler()
model = make_pipeline(transformer, clf)

# TASK: Fit the pipeline on the training set
model.fit(X_train, y_train)

# Cross-validation
# cv = ShuffleSplit(n_splits=5, test_size=.4, random_state=0)
# scores = cross_val_score(model, features, label, cv=cv)
# print scores
# print 'Crossval score: %0.2f +/- %0.2f ' % (scores.mean(), scores.std())

# TASK: Predict the outcome on the testing set.
# Store the result in a variable named y_predicted
y_pred = model.predict(X_test)
print 'Test accuracy: %.4f' % accuracy_score(y_test, y_pred) 

# TASK: Print the classification report
print classification_report(y_test, y_pred)

# TASK: Print the confusion matrix
print '=======confusion matrix=========='
cm = confusion_matrix(y_test, y_pred)

plot_learning_curve(model, features, label)
#plot_confusion_matrix(cm, feature_names, normalize=False)
plt.show()

# TASK: Is the score good? Can you improve it changing
#       the parameters or the classifier?
#       Try using cross validation and grid search

# TASK: Use dill and gz to persist the trained model in memory.
#       1) gzip.open a file called my_model.dill.gz
#       2) dump to the file both your trained classifier
#          and the target_names of the dataset (for later use)
#    They should be passed as a list [clf, dataset.target_names]

import dill 
import gzip
with gzip.open('my_model.dill.gz', 'wb') as f:
    dill.dump([model, dataset.target_names], f)
    
    