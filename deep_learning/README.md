# Deep learning notes

## Regularization

### Shapes
* L1 regularization: `\/` absolute value because `l1 = C + (|w1| + |w2| + ...)`
* L2 regularization: `(_)` polynomial because `l2 = C + (w1^2 + w2^2 + ...)`

### Types
* You can have weight regularizers or activations (i.e. output of a layer) regularizers.

### Intuition
* L1 regularization will have a tendency to push the weights to 0 (see shape). Because of that **L1 regularization is often used for feature selection** (the non-zero weights that remain are the most prominent features)
* In general for sparse input, L2 works best because it does not push the weights to 0 as much. A bit smoother.

## Gradient

### Computation
The gradient can be computed:
* at the end of the mini-batch (on the mini batch); 
* at the end of the full epoc; 
* on each input point

### Mini-batch size
* Using mini-batches converges faster than GD computed at the end of an epoch.
* The bigger the mini-batch size, the more noisy the search for the minimum.
* SGD (stochastic gradient descent) computed on each point is equivalent to having batch_size=1 in Keras.
* Adaptive (learning rate) gradient techniques like `adagrad` are better but more computationally expensive.


## Data augmentation
* Rescaling, translation, rotation, scaling, distortion, filtering (invert 
colors, blur), add noise, obstruct parts of the picture.
* Think of any transformation where  a human would still recognize the thing.
* Augmenting the data can help you to avoid the wolf VS husky problem.

## Generating data
* Example: take a license plate and use many different backgrounds (that 
have no license plate on it). So you create a bunch of new training pictures
 * Use an established data generation technique (deterministic) like 
 text-to-speech to generate sounds for your voice to text classifier

## Matplotlib tip
* Use `%matplotlib inline` to plot stuff inline in the notebook.
* Use `%matplotlib notebook` to get interactive charts in the notebook.

## Questions
* Any DL architecture where there is feedback?