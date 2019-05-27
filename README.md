### Post-synaptic potential

This repository contains the PyTorch implementation for the post-synaptic potential (PSP) regularization technique.
All the differentiation is performed by autograd: the post-synaptic potentials are passed along with the overall output of the network, during the forward propagation step.

The following models are included:
0. LeNet-5 for MNIST and Fashion-MNIST
0. ResNet-18 for CIFAR-10
0. MobileNetv2 for CIFAR-10
0. All-CNN-C for CIFAR-10
