# mnist-from-scratch

## Purpose

This was a learning exercise to understand the fundamentals of the foundation of modern deep learning models, inspired by the 3Blue1Brown [series on how neural networks work.](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=QZkHyVKURtvLmyD1) This was built entirely from scratch using just the NumPy Python library in an effort to maximize my intuition on how every part of the perceptron would be implemented.

This involved writing:
 - The necessary activation functions
 - A forward propagation function
 - Fundamental backpropagation logic
 - Randomization of initial weights
 - Tests involving multiple epochs of training

The main perceptron Jupyter notebook explains every part of the code in as much detail as possible, even if some of it should be self explanatory or involves fairly basic calculus. This was done to make this notebook extremely easy to revisit at any point and understand not only how the code implementation works but also all of the mathematical and architectural intuition one would need to understand.

## Pytorch Version

Additionally, this project also has a more compact Pytorch implementation. I wrote this after the fact so I could understand exactly how much effort modern libraries save compared to doing everything from the ground up, as well as the runtime performance difference between the NumPy version which runs single-threaded on the CPU and the Pytorch version meant to run on a CUDA-enabled GPU. For a neural network this small, it turned out that the run time difference was negligible, but the code itself did turn out to be more compact.

## Dependencies

This project only has two dependencies: NumPy and Pytorch 2.5.1 with CUDA 12.4. 

