# Advanced Deep Learning Data Weekends
TODO
* [ ] 1c_logistic_regression
* [ ] 1d_fully_connected
* [ ] 2a_convolutional_neural_networks

Notes;
* [EWMA](https://en.wikipedia.org/wiki/EWMA_chart)
* Compute the derivative of the cost function.
* Gradient = Evaluate derivative at a point (or all the points in a mini-batch (also called batch) and average it)
* The gradient gives you in what direction to move.
*  SGD + Momentum = gradient EWMA. The gradient retains some of the speed with wich it was moving.
 * RMSPROP: take EWMA of the gradient. The only difference with the momentum method is that the learning rate is not fixed. It's modulated by the gradient.
 
 Dimensions in CNNs:
 * `height` = `n_rows`
 * `width` = `n_columns`
 * 4D input = [`batch_size`, `height`, `width`, `n_input_channels`]
 * Filters = [`num_height`, `num_width`, `n_input_channels`, `n_output_channels` ]
 * Result of convolution = [`batch_size`, `height`, `width`, `n_output_channels`]
 
 
 Tensorboard:
 * Keras callbacks. You can use TensorBoard.
 * With tensorboard you can visualize embeddings in the intermediate layers. Or images.
 
 Alternative to ML for image processing
 * Open VC to detect faces.