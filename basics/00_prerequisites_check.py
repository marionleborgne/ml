# this script just imports the packages
# we need and checks the versions are correct

import pandas
import numpy
import bokeh
import sklearn
import matplotlib
import scipy
import flask

assert(pandas.__version__ >= '0.19.0')
assert(numpy.__version__ >= '1.11.2')
assert(bokeh.__version__ >= '0.12.3')
assert(sklearn.__version__ >= '0.18')
assert(scipy.__version__ >= '0.18.1')
assert(matplotlib.__version__ >= '1.5.3')
assert(flask.__version__ >= '0.11.1')

print "Houston, we are go! :)"
