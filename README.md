# Deep Learning Nanodegree Udacity
This is the repository for my implementations on mayor projects of the Deep Learning Nanodegree from Udacity.

[Syllabus](https://www.udacity.com/course/deep-learning-nanodegree--nd101)

## Neural Networks.
* [Weight Initilization review](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Neural%20Networks/Weight%20Initialization/weight_initialization.ipynb): 
  * Implication of different initializations over Cost function and Gradient descent. 
  * Reviewed:
    * Ones initializatialization.
    * Uniform distribution, saled uniform.
    * Normal distribution, truncated distribution.
    * Comparison to Xavier initialization.     
    
* [Sentiment Analysis using MLPs](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Neural%20Networks/Sentiment%20Analysis%20MLP/Sentiment_Classification_Projects.ipynb): 
  * Predict Positive/Negative sentiment over movie reviews.
  * Preprocess data:
    * Create vocabulary, word frequency.
    * Analyze word-freq/sentiment review ratio.
    * Bit encoding per word.
  * Built the neural network.
  * Reviewed limitations with word freq instead of word-sentiment relationship. 10% Validation accuracy improvement.

* [Bike Sharing Project](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Neural%20Networks/Bike%20Sharing%20MLP%20model/Your_first_neural_network.ipynb): 
  * Load & prepare the data: 
    * Normalize features.
    * Created training, validation and test data.
  * Implement forward and backward propagation.
  * Trained and tested accuracy.

## Convolutional Neural Networks.
* [CNN Autoencoder](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Convolutional%20Neural%20Networks/CNN%20Autoencoder/Convolutional_Autoencoder.ipynb): 
  * Usage of CNNs for encoding-decoding.
  * Denoising images.

* [Data Augmentation & Transfer Learning](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/tree/master/Convolutional%20Neural%20Networks/Data%20augmentation%20%26%20Transfer%20Learning): 
  * Explored data augmentation of CIFAR-10 with ImageDataGenerator from Keras, and impact of it over training.
  * Reviewed transfer learning on VGG-16, bottleneck feature extraccion and new FC layers over them.

* [Dog Breed Prediction Project](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Convolutional%20Neural%20Networks/Dog%20Breed%20Project/dog_app.ipynb): 
  * Created CNN model from scratch and achieved at least 5% test accuracy in the first 5 epochs using data augmentation.
  * Used transfer learning of Xception model, and data augmentation to achieve 83% test accuracy. 
  * Xception paper: [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
  
 ## Recurrent Neural Networks.
 * Character-Level LSTM Network:
  
 ## Generative Adversarial Neural Networks.
 
 ## Deep Reinforcement Learning.
