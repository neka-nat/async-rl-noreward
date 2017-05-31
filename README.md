### Variation of Asynchronous RL in Keras (Theano backend) + OpenAI gym [A3C]
This is a simple variation of [asynchronous reinforcement learning](http://arxiv.org/pdf/1602.01783v1.pdf) written in Python with Keras (Theano backend). Instead of many threads training at the same time there are many processes generating experience for a single agent to learn from. 

### Explanation
There are many processes (tested with 4, it should work better with more in case of Q-learning methods) which are creating experience and sending it to the shared queue. Queue is limited in length (tested with 256) to stop individual processes from excessively generating experience with old weights. Learning process draws from queue samples in batches and learns on them. In A3C network weights are swapped relatively fast to keep them updated.

### Currently implemented and working methods
* [n-step Q-learning](https://github.com/Grzego/async-rl/tree/master/q-learning-n-step)
* [A3C](https://github.com/Grzego/async-rl/tree/master/a3c)

### Requirements
* [Python 2.7](https://www.python.org/downloads/)
* [Keras](http://keras.io/)
* [Theano](http://deeplearning.net/software/theano/) ([Tensorflow](https://www.tensorflow.org/) would probably work too)
* [OpenAI](https://gym.openai.com/)
