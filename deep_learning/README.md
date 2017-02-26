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

## Hidden layers
* Choosing a number of hidden neurons that is the same order of magnitude as the input is good rule of thumb. To avoid too much compression.

## CNN

### Filtering
* Convolution: filter. If image is 9x9, then filtered image is 7x7 if no padding. Add padding (like zero-padding) in the input image to preserve the original image size.
* The filters are initially random. But the CNN learns over time what filters work best.
* It doesn't make a lot of sense to make very large filters. In object recognition at least. 
* It's better to go deeper (more layers of filters) than to have few very large filters.
* That is why it can be useful to use a retrained CNN like inception.
* Filter is also called feature.

### Normalization
* Use reLU before pooling to normalize. I.e. get rid of the negative numbers

### Pooling
* Pooling strides can be overlapping or not.
* Helps the system to generalize

### Architecture
* Conv layers (Filter + relu) recognize patterns.
* Pooling layers forget the _exact_ location of the pattern.
* You can alternate depending on what you want to do. E.g: `F > R > F > R > Pooling > F > R > Fully connected layer(s) > Softmax`
* before the fully connected layer(s) you need to flatter the output of the convolution layers.

### Hyperparameters

#### Convolution
* Size and shape of features. The pixels of the features can be next to each other or far from each other. Use features with pixels that are far from each other: when you want to capture objects taht are far apart.
* Stride size

### Pooling
* Window size
* Type of pooling
* It's not mandatory. In some case you don't want pooling if you don't want to bakin in forgetfulness in you network.
* Examples: cat VS dog -> use pooling. If you care about spatial/positional details, don't use pooling. For example in a chess game.

### Tensor math
* Channels (the 3rd dim of the cube): 1 channel for grey scale, 3 for RGB.
* Notation: `(x)` is conv
* Synomyms: output_size = num_filters = num_channels

#### Grey scale example

* The first filter is 3x3, the input size is 1 (because grey scale), number of filters of this conv layer is 10.
* The second filter is 2x2, the input size is 10 (because 10 filters), number of filters of this conv layer is 32.
* Pooling does not have channels.

```
(28,28,1) (x) (3,3,1,10)  
= (28,28,10) (x) (2,2,10,32)
= (28,28,32)
```

#### RGB example
* The first filter is 3x3, the input size is 3 (because RGB), number of filters of this conv layer is 10.
* The second filter is 2x2, the input size is 10 (because 10 filters), number of filters of this conv layer is 32.
* Pooling does not have channels.

```
(28,28,3) (x) (3,3,3,10)  
= (28,28,10) (x) (2,2,10,32)
= (28,28,32)
```

### Use cases
* CNN only useful when you have an input where `neaby elements are correlated in space & time`.
* For time what you can do is add a time dimension (in addition of the `num_channels` dim)
* An example use for learning time correlation in speech (for speech synthesis) -> `wavenet`
* Wavenet is a 1D CNN with the time dimension as input (double check that)
* By convention, if you want to add time in your CNN, then you would add the time window size after the number of inputs
```
num_records = X.shape(0)
X = X.reshape(num_records, time_window_size, 28, 28, num_channels) # for 28 x 28 images
```

### Tensorflow VS Theano backend in Keras
* CNN tf ordering: height, width, num channels
* CNN theano ordering: num channels, height, width

## Matplotlib tip
* Use `%matplotlib inline` to plot stuff inline in the notebook.
* Use `%matplotlib notebook` to get interactive charts in the notebook.

## Questions
* Any DL architecture where there is feedback?