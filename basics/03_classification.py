import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# load the data
df = pd.read_csv('data/isp_data.csv')

# define features (X) and target (y)
X = df[['download', 'upload']]
y = df['label']

# create an instance of the decision tree model class
model = DecisionTreeClassifier(random_state=0)  # 1) can change params here

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=0)

# train the model
model.fit(X_train, y_train)

print "\nClassification Score: %0.3f" % model.score(X_test, y_test)

# display the confusion matrix
y_pred = model.predict(X_test)
print "\n=======confusion matrix=========="
print confusion_matrix(y_test, y_pred)


# # Exercises
#
# 1) try to improve the score changing the parameters
#    at the initialization of the DecisionTreeClassifier
#    things you can try:
#    - set the max_depth of the tree
#    - set the min_samples of the tree
#    - look in the documentation
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
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
