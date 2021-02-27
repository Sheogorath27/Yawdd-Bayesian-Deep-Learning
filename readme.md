
# Quantifying Uncertainty in Deep Learning for Safety-Critical Tasks 

This repository contains some of the experiments conducted for my MSc Thesis. I will upload the complete work after publishing.

## Introduction

This work explores the advantages of Bayesian Deep Learning and robust Machine Learning. We use the YawDD dataset to train models to detect drowsy drivers. The SOTA solutions for this problem use deep learning models and have very high accuracy scores. The possibility of demonstrating the need for robustness and uncertainty even at these high scores was our main reason for selecting this dataset.
- We match accuracy scores of previous works with a computationally much smaller model. We use Lenet-5 while pre-trained GoogleNet and Inception-V3 were used in earlier works. This is very significant because the model use-case involves high-frequency data and limited hardware capabilities.
- We use uncertainty to highlight the problem of the mislabeled dataset in the previous training methodology. We propose a new sampling method to mitigate the problem and this method can be used in other video datasets.
- We show the unreliability of previous models with out-of-distribution data and use uncertainty to decrease false-negative cases(drowsy drivers classified as not drowsy) keeping in mind the safety-critical nature of the application

## Requirements

All experiments were conducted in the Google Colab environment and iPython notebooks contain instructions to modify default settings wherever needed. 

## File Structure

The files and scripts for reproducing the experiments are organized as follows:

```
.
+-- dataSampler.py: Functions in this file are used to prepare different datasets.
+-- models.py: file contains the code for all the models used in the experiment.
+-- concreteLayers.py: Script contains concrete dropout layers classes.
+-- analysis.py: Script has functions to calculate and visualize uncertainty measures.
+-- prepare dataset.ipynb: Notebook contains examples that demonstrate the use of functions in the dataSampler file.
+-- training.ipynb: Examples in this notebook show model training.
+-- model_analysis.ipynb: examples of model analysis based on uncertainty measures.
+-- sample_thesis.pdf: Initial version of my thesis, containing background study and some experiment design.
```

## References for Code Base
The code for Concrete layers is adapted from [this GitHub repo](https://github.com/yaringal/ConcreteDropout).