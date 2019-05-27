### Post-synaptic potential

This repository contains the PyTorch implementation for the post-synaptic potential (PSP) regularization technique.
All the differentiation is performed by autograd: the post-synaptic potentials are passed along with the overall output of the network, during the forward propagation step.

The following models are included:

* LeNet-5 for MNIST and Fashion-MNIST
* ResNet-18 for CIFAR-10
* MobileNetv2 for CIFAR-10
* All-CNN-C for CIFAR-10
