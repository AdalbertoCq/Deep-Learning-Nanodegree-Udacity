# Deep Learning Nanodegree Udacity
This is the repository for my implementations on mayor projects of the Deep Learning Nanodegree from Udacity.

[Syllabus](https://www.udacity.com/course/deep-learning-nanodegree--nd101)

## Neural Networks.
* Mathematical demonstrations:
  * [Backpropagation](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Neural%20Networks/backprop.PDF)
  * [Softmax & Cross-Entropy gradients](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Neural%20Networks/cross_entropy_softmax.PDF)
  * [Batch Normalization gradient](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Neural%20Networks/batch_norm_backprop.PDF)
  
* [Weight Initilization review](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Neural%20Networks/Weight%20Initialization/weight_initialization.ipynb): 
  * Implemented using TensorFlow.
  * Implication of different initializations over Cost function and Gradient descent. 
  * Reviewed:
    * Ones initializatialization.
    * Uniform distribution, saled uniform.
    * Normal distribution, truncated distribution.
    * Comparison to Xavier initialization.     

* [Batch Normalization](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Neural%20Networks/Batch%20Normalization/Batch_Normalization_Exercises.ipynb):
  * Implemented on TensorFlow.
  * Used in fully connected and convolutional layers.
  * Two levels of implementation:
      * Higher level of abstraction, tf.layers.batch_normalization: TensorFlow takes care of the normalization for training and inference, control dependencies through tf.control_dependencies() and tf.GraphKeys.UPDATE_OPS.
      * Lower level, tf.nn.batch_normalization: Explicit implementation instanciating gamma, beta, and calculating the batch/population mean, variance. Control training and inference through tf.cond().
  
* [Sentiment Analysis using MLPs](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Neural%20Networks/Sentiment%20Analysis%20MLP/Sentiment_Classification_Projects.ipynb): 
  * Implemented on Numpy/Python.
  * Predict Positive/Negative sentiment over movie reviews.
  * Preprocess data:
    * Create vocabulary, word frequency.
    * Analyze word-freq/sentiment review ratio.
    * Bit encoding per word.
  * Built the neural network.
  * Reviewed limitations with word freq instead of word-sentiment relationship. 10% Validation accuracy improvement.

* [Bike Sharing Project](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Neural%20Networks/Bike%20Sharing%20MLP%20model/Your_first_neural_network.ipynb): 
  * Implemented on Numpy/Python.
  * Load & prepare the data: 
    * Normalize features.
    * Created training, validation and test data.
  * Implement forward and backward propagation.
  * Trained and tested accuracy.

## Convolutional Neural Networks.
* [CNN Autoencoder](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Convolutional%20Neural%20Networks/CNN%20Autoencoder/Convolutional_Autoencoder.ipynb): 
  * Implemented using Keras.
  * Usage of CNNs for encoding-decoding.
  * Denoising images.

* [Data Augmentation & Transfer Learning](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/tree/master/Convolutional%20Neural%20Networks/Data%20augmentation%20%26%20Transfer%20Learning): 
  * Implemented using Keras.
  * Explored data augmentation of CIFAR-10 with ImageDataGenerator from Keras, and impact of it over training.
  * Reviewed transfer learning on VGG-16, bottleneck feature extraccion and new FC layers over them.

* [Dog Breed Prediction Project](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Convolutional%20Neural%20Networks/Dog%20Breed%20Project/dog_app.ipynb): 
  * Implemented using Keras.
  * Created CNN model from scratch and achieved at least 5% test accuracy in the first 5 epochs using data augmentation.
  * Used transfer learning of Xception model, and data augmentation to achieve 83% test accuracy. 
  * Xception paper: [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
  
## Recurrent Neural Networks.
* Mathematical demonstrations:
  * [RNN Backpropagation through time](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Recurrent%20Neural%20Networks/rnn_through_time_backprop.pdf)
  * [LSTM Backpropagation through time](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Recurrent%20Neural%20Networks/lstm_through_time_backprop.pdf)
  * [GRU Backpropagation through time](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Recurrent%20Neural%20Networks/gru_through_time_backprop.pdf)
  
* [Character-Level LSTM Network](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Recurrent%20Neural%20Networks/Character%20wise%20LSTM/Anna_KaRNNa.ipynb): 
  * Implemented in TensorFlow.
  * Developed a Character-Wise RNN sequence predictor. A two 2 layer depth LSTM with Tx=50 time sequence length. With a 128 dimension for the LSTM memory cell, and a vocabulary size 83.
  * Steps:
    * Data processing for minibatches. 
    * Built LSTM model.
    * Optimizer & Gradient clipping.
    * Checkpoint training. 
    * Sequence generation with output sampling.    
 
* [Embeddings and Word2vec](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Recurrent%20Neural%20Networks/Embeddings%20and%20Word2vec/Skip-Gram_word2vec.ipynb): 
  * Implemented in TensorFlow.
  * Implemented and trained a Skip-gram Word Embedding matrix.
  * Used Subsampling, negative sampling.
  * Visualization of word vectors using T-SNE.
  * Based on papers:
    * [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
    * [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
  
* [Sentiment Prediction](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Recurrent%20Neural%20Networks/Sentiment%20Prediction/Sentiment_RNN.ipynb): 
  * Implemented in TensorFlow.
  * Sentiment prediction using Word Embedding on LSTM.

* [The Simpsons Script Generation](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Recurrent%20Neural%20Networks/Simpsons%20TV-Script%20generation/dlnd_tv_script_generation.ipynb): 
  * Implemented in TensorFlow.
  * Language sequence generation on a LSTM network using Word Embedding.

## Generative Adversarial Neural Networks.
* [GAN Personal Notes](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Generative%20Adversarial%20Networks/GAN%20notes.pdf)
* [GAN over MNIST db](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Generative%20Adversarial%20Networks/GAN%20MNIST/Intro_to_GANs_Exercises.ipynb): 
  * Implemented in TensorFlow.
  * GAN implementation for the MNIST database.
  * [Generative Adversarial Networks Paper](https://arxiv.org/pdf/1406.2661.pdf)
 
* [DCGAN](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Generative%20Adversarial%20Networks/DCGAN%20SVHN%20/DCGAN.ipynb): Deep Convolutional GAN: 
  * Implemented in TensorFlow.
  * DCGAN implementation for the Street View House Number database.
  * [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks Paper](https://arxiv.org/abs/1511.06434)
 
## Deep Reinforcement Learning.
* [Reinforcement Learning Personal Notes](Placeholder)
* [Frozen Lake](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Reinforcement%20Learning/Frozen%20Lake/Dynamic_Programming.ipynb):
  * Implementation on Frozen Lake enviroment.
  * Reinforcement Learning by Richard S. Sutton and Andrew G. Barto: Chapters 3 & 4
  * Covers Finite Markov Processes and Dynamic Programming:
    * Policy Evaluation.
    * Policy Improvement.
    * Policy Iteration. 
    * Truncated Policy Evaluation.
    * Value Iteration.
* [BlackJack](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Reinforcement%20Learning/BlackJack/Monte_Carlo.ipynb):
  * Implementation on BlackJack enviroment.
  * Reinforcement Learning by Richard S. Sutton and Andrew G. Barto: Chapter 5.
  * Monte Carlo Methods:
    * Monte Carlo Predictions: State-value and Action-value functions.
    * Monte Carlo Control.
    * GLIE MC Control(Greedy in the limit with Infinite Exploration).
    * Constant aplha-GLIE MC Control.
* [CliffWalking](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Reinforcement%20Learning/CliffWalking/Temporal_Difference.ipynb):
  * Implementation on CliffWalking enviroment.
  * Reinforcement Learning by Richard S. Sutton and Andrew G. Barto: Chapter 6.
  * Temporal-Difference Methods:
    * Temporal-Difference Predictions: State-value and Action-value functions.
    * Sarsa.
    * Q-Learning (Sarsamax).
    * Expected Sarsa.
* [Taxi-v2](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/tree/master/Reinforcement%20Learning/taxi-v2):
    * Implemented agent to solve the OpenAI gym of [Taxi](https://gym.openai.com/envs/Taxi-v2/).
    * Tested Q-Learning, Sarsa, Expected Sarsa.
    * Best Score over 100 episode average rewards: 9.359 on Q-Learning.
* Reinforcement Learning in cotinuous spaces:
  * [Discretization](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Reinforcement%20Learning/RL%20in%20continuous%20spaces/Discretization.ipynb)
  * [Tile Coding](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Reinforcement%20Learning/RL%20in%20continuous%20spaces/Tile_Coding.ipynb)
* [Deep Q-Learning](https://github.com/AdalbertoCq/Deep-Learning-Nanodegree-Udacity/blob/master/Reinforcement%20Learning/Deep%20Q-Learning%20Cart-Pole-v0/Q-learning-cart_own_implementation.ipynb):
  * Deep Q-Learning implementation.
  * Implementations of Neural Network Action-Value approximator in TensorFlow and Experience Replay memory.
  * Solved [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/): "CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials."

