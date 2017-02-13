# Machine Learning @ DataWeekends

## Exercises
* [x] `01a_multiple_plots.py`
* [ ] `01b_plot_csv.py`
* [ ] `advanced`
* [x] `02_regression.py`
* [x] `03_classification.py`

# Notes

### `01b_plot_csv.py`
* See usage of `pd.groupby()` to plot categories.

###`02_regression.py`
* Tip to add a dimension: `a[:, np.newaxis]`` is like `[[h] for h in a]`` 
* Check out residual sum of squares in the demo script.
* Plot best fit line in N dimensions by projecting the regression 
hyperplane on each feature axis.
* Remember, don't shuffle the data if the data is temporal. If it's not, 
then it's important to shuffle it. 
* Explained variance `R^2`. Also called goodness of fit. 
* See [Anscombe's quartet](https://en.wikipedia.org/wiki/Anscombe's_quartet)
* Check [this](http://seaborn.pydata.org/tutorial/regression
.html#plotting-a-regression-in-other-contexts) out to plot best fit and data 
distribution.

### `03_classification.py`
* Precision and Recall are terms that are only used in binary classification 
problems and they are relative to a specific class.
* Binary classifier can be used in a multi-class problem -> One class VS all.
* Accuracy: how many times was I correct? `(tp+tn)/total`
* Precision: how many times was I correct when I predicted class 1. 
`tp/predicted_1`
* Recall: how many do I detect 1 out of all the class 1 population.  
`tp/actual_1`
* f1_score=(p.r)/(p+r) 
* Example

```
In this example:
  class 0: 99%
  class 1: 1%
An accuracy of 99 % is actually bad or not informative. Because precision 
could be 0.
```
* Instead of using grid search use `BayseOptimizer`. Better hyper param 
optimizer. 

#### `04_clustering.py`
* Metric to measure how good the clustering is: `silouette`


#### `05_data_preprocessing.py`
* Data normalization. Look at `MinMaxScaler` and `StandardScaler` (mean of 0
 and standard deviation of 1). Examples [here](http://machinelearningmastery.com/prepare-data-machine-learning-python-scikit-learn/)
* `pd.get_dummies()` will convert a DF column from categories to binary
 
 
 #### `06_accuracy.py`
 * Interactive [scikit-learn cheat sheet](http://scikit-learn.org/stable/tutorial/machine_learning_map/)
 




