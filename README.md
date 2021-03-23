# **Behavioral Cloning** 
In this project, I trained a model to predict steering values based on images received from a self-driving simulator. Using the model, the car was able to navigate a complete simulator track autonomously. 

[//]: # (Image References)

[architecture]: ./docs/model.png "Model Visualization"
[aug1]: ./docs/aug1.png "data augmentation"
[aug2]: ./docs/aug2.png "data augmentation"

## Model Architecture
For the model, I drew inspiration from an NVIDIA Convolutional Neural Network described in this [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). After considering the size and complexity of my dataset, to avoid over-fitting, for this use case I settled on getting rid of some of those layers. I made use of additional dropout layers to also prevent overfitting.  My final model consisted of the 12 layers described below:

![summary of network structure][architecture]

## Training Details
I made use of training data gotten after driving for about two laps on the test track simulator. When driving, I tried to ensure the car remained at the middle lane and added few examples of the moving back to the lane in conditions when its swerves.

### Data Augmentation
Due to the small size of my dataset, I applied some data augmentation techniques to ensure the model was generalizing appropriately. Some of those techniques include:

* Stochastically selecting between camera images from the center, left, and right cameras and issuing corrections to steering value based on position of image.
![Images from left center and right camera][aug1]
* I also vertically flipped images uniformly to help augment the data
![Original and Vertically Flipped Images][aug2]

### Parameters

* Optimizer: [Adam Optimizer](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/AdamOptimizer) with a default learning rate of 0.001.

* Loss Function: [mean squared error](https://www.tensorflow.org/api_docs/python/tf/keras/losses/MSE).

* Batch Size: Default value of 32.

* Epochs: 10.