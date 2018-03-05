# Receptive-Field-in-Pytorch
Numerical Computation of Receptive Field in Pytorch
I present a simple pytorch code that computes numerically the Receptive Field (RF) of a convolutional network. It can work with very complicated networks in 2D, 3D, with dilation, skip/residual connections, etc.
In the Jupyter notebook I explain how can we compute the RF both analitycally and numerically. I show some code that computes both.

In the python file a simple function to compute it.

## Requeriments
First you must change the max pooling layers of your network by average pooling and turn off any batchnorm and dropout that you might have. This is in order to avoid sparse gradients (More detailed explanation in the Jupyter Notebook).

You must provide also an numpy array that will be filled with ones and with the appropiate shape for your specific network.
