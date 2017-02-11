import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split, learning_curve


# Load the data
df = pd.read_csv('./data/new_titanic_features.csv')


# Create Features and Labels
features = df[['Male', 'Family',
               'Pclass2_one', 'Pclass2_two', 'Pclass2_three',
               'Embarked_C', 'Embarked_Q', 'Embarked_S',
               'Age2', 'Fare3_Fare11to50', 'Fare3_Fare51+', 'Fare3_Fare<=10']]

label = df['Survived']


# Train test split
x_train, x_test, y_train, y_test = train_test_split(features,
                                                    label,
                                                    test_size=.2,
                                                    random_state=0)


# Create instance of Logistic Regression classifier
model = LogisticRegression(C=1, random_state=0)
lrc = model.fit(x_train, y_train)
pred_train = lrc.predict(x_train)

print
print 'Training Accuracy:', metrics.accuracy_score(pred_train, y_train)

pred_test = lrc.predict(x_test)
print
print 'Test Accuracy:', metrics.accuracy_score(y_test, pred_test)
print
print 'Confusion Matrix:'
print metrics.confusion_matrix(y_test, pred_test)
print
print 'Classification Report:'
print metrics.classification_report(y_test, pred_test)


# Get feature importance
coeffs = pd.DataFrame(zip(features.columns, lrc.coef_.ravel()),
                      columns=['feature', 'coefficients'])
print "Feature importances:", coeffs


# Cross-validation
cv = model_selection.ShuffleSplit(n_splits=5, test_size=.4, random_state=0)
scores = model_selection.cross_val_score(model, features, label, cv=cv)
print scores
print 'Crossval score: %0.2f +/- %0.2f ' % (scores.mean(), scores.std())


# Learning Curve
tsz = np.linspace(0.1, 1, 10)
train_sizes, train_scores, test_scores = learning_curve(lrc, features, label,
                                                        train_sizes=tsz)

fig = plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), 'ro-', label="Train Scores")
plt.plot(train_sizes, test_scores.mean(axis=1), 'go-', label="Test Scores")
plt.title('Learning Curve: Logistic Regression')
plt.ylim((0.5, 1.0))
plt.legend()
plt.draw()
plt.show()


# Exercises
#
# 1) Try rescaling the age feature with the standard scaler.
#    Does that change the result?
#    http://scikit-learn.org/stable/modules/preprocessing.html
#
# 2) Experiment with another classifier for example
#    DecisionTreeClassifier or SVC
#    you can find classifiers here:
#    http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
#    Which ones are impacted by the age rescale? Why?
