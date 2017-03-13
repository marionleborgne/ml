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

---

# Deep learning on AWS

This is a very brief guide to getting setup for deep learning with Tensorflow and Keras on an AWS GPU instance. It walks through setting up an instance from an AMI (Amazon Machine Image) that has Tensorflow installed already -- this is the hard part and it is already done. Most of this guide is about setting up IPython notebook so that you can conveniently experiment in your browser.

## Tensorflow AMI

The AMI is based on this AMI provided by Stanford: http://cs231n.github.io/aws-tutorial/

It has CUDA 7.5 and CuDNN 4.0 installed, along with Tensorflow 0.9, Keras 1.0.5, and all other dependencies for the notebooks already installed in an Anaconda python environment. After launching the instance, the environment can be activated by running

```bash
source activate tensorflow 
```

The AMI id is ami-97ba3a80.

## Sign up

First you need to sign up for AWS at https://aws.amazon.com/, enter payment information, etc. Don't worry, you will only pay for what you use.

## Launch instance

You can launch this AMI by going to https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#LaunchInstanceWizard:ami=ami-97ba3a80

This should give you a list of instance types to choose from. We want the g2.2xlarge GPU instance, so select it. After that, click the Next button at the bottom of the page to get to "Step 3: Configure Instance Details".

Here you can change your "Purchasing option". If you select "Request Spot instances", you can enter a bid price for the instance. This means that it may take longer for you to get the instance and the instance could be shutdown at an arbitrary time, but you can usually get an instance for much cheaper than the On Demand price of $0.65/hr. You should see the current Spot price for different regions after (the higher you max price, the faster you will get the instance and the more likely you are to retain the instance). I would recommend a Spot instance for simple experimentation and On Demand when you are sure you want to retain the instance for more serious work.

Clicking next again will get you to "4. Add Storage". Here you can choose to attach EBS volumes to the instance if you have any. Additionally, you can choose to retain the boot volume when the instance terminates by unchecking the "Delete on Termination" option. Note that retained volumes will be subject to EBS pricing.

After selected your Purchasing option, skip ahead to step "6. Configure Security Group" at the top of the screen. Here we will add 2 security rules (if you already have a security group defined, you can choose an existing security group instead).
1. Click "Add Rule" and select "HTTPS" as the Type
2. Click "Add Rule" again, check that Type is "Custom TCP Rule", enter 8888 as the port, and select "Anywhere" as the Source

Click "Review and Launch" at the bottom of the page. You can review your configuration, then click "Launch". A dialog will appear to create or select a key pair. If you need to, create a key pair and download the .pem file to a safe place on your computer.

After requesting the instance, you should see a view of the AWS console where you can see the status of your instance. After the instance is ready, we can log in!

## Logging into the instance

First, find your .pem file and restrict the permissions on it:
```
chmod 400 pem_file_name.pem
```

Get the Public DNS URL of your instance in the AWS console (mine was ec2-52-23-241-12.compute-1.amazonaws.com), and ssh to the host:

```
ssh -i pem_file_name.pem ubuntu@ec2-52-23-241-12.compute-1.amazonaws.com
```

## Install needed packages
```
sudo apt-get install h5py
sudo pip install keras
```
## Install and set up Jupyter

First, let's install Jupyter:
```
sudo pip install jupyter
```

In order to use our instance via IPython notebook, we need to create a self-signed SSL certificate so your browser can securely connect to IPython notebook on the instance. The following will start a wizard to create the certificate - answer the questions as well as you can.

```
mkdir certs && cd certs
sudo openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout mycert.pem -out mycert.pem
```

Let's also generate a password to use to log in. Start `ipython` and run
```python
from IPython.lib import passwd
passwd()
```

Copy and save the sha1 string that was generated for later and exit IPython. We need to configure IPython notebook. The following will create a `~/.jupyter/jupyter_notebook_config.py` file.
```
jupyter notebook --generate-config
```

Let's change the default config to use our new certificate and password:
```
cd ~/.jupyter/
nano jupyter_notebook_config.py
```

Put the following at the top of the file, substituting in the password sha1 and certificate path, and save the file.
```
c = get_config()

c.IPKernelApp.pylab = 'inline'

c.NotebookApp.certfile = u'/home/ubuntu/certs/mycert.pem' #location of your certificate file
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.password = u'sha1:68c136a5b064...'  #the encrypted password we generated above
c.NotebookApp.port = 8888
```

You can now start the IPython notebok by running `jupyter notebook`. You should now be able to navigate to https://<your-instance-public-dns>:8888 in your browser (don't forget the https://). You will likely get a warning that the certificate can't be verified. This is as expected, just proceed anyways. Then you will have an opportunity to enter your password for IPython notebook.

Much of the preceding setup came from http://blog.impiyush.me/2015/02/running-ipython-notebook-server-on-aws.html

## Install AWS CLI:

You can do almost everything you need on AWS on the commandline with the AWS CLI without having to do things with the AWS console page. For more information on installation and usage, refer to Lot of details at http://docs.aws.amazon.com/cli/latest/userguide/installing.html.


