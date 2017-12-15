# Implementation and experimentation of compression for variational Bayesian neural networks
=======
This repository implements supervised Bayesian neural networks according to `Blundell et al.' (2015) <https://arxiv.org/abs/1505.05424>`_. The BNNs are then compressed into a smaller network via a decision function.

The models are implementated in `TensorFlow  1.3 <https://www.tensorflow.org/api_docs/>`_.


Installation
------------
Please make sure you have installed the requirements before executing the python scripts.


**Install**


.. code-block:: bash

  pip install scipy
  pip install numpy
  pip install matplotlib
  pip install tensorflow(-gpu)
