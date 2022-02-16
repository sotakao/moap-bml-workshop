# Gaussian Process Tutorials for MOAP Workshop

This repo contains the tutorials that I made in companion to my presentation at the UCL-MOAP workshop on "Bayesian Machine Learning for Weather and Climate". In particular, you can find implementations of all the experiments shown in the presentation and further details on various techniques that I didn't have time to present at the workshop.

The tutorial is divided into 5 parts, which cover the following topics.
1. Basics of Gaussian process regression including how to set up a model, how to generate samples, how to condition on data and how to optimise for the hyperparameters using the log marginal likelihood.
2. Sparse GPs, in particular Titsisas' inducing point method. The notebook contains information on the mathematical derivation of the model, comparative experiments with a vanilla Gaussian process and how to train the model.
3. Gaussian process inference for timeseries data using Kalman filtering or smoothing techniques.
4. Global temperature interpolation using Matérn GPs defined on the sphere and comparison with a vanilla GP regression.
5. Global wind speed interpolation using projected vector kernels.

## Requirements
To run the first three notebook, please install the packages in the `requirements.txt` file by running
```
pip install -r requirements.txt
```
In addition, the codes in the fourth and fifth tutorials use our `riemannianvectorgp` package (can be found [here](https://github.com/MJHutchinson/ExtrinsicGaugeIndependentVectorGPs/tree/main/riemannianvectorgp)), which is not on pip so you have to clone this into your repository and install the requirements therein.

## Data
All the datasets used in the tutorials are saved in the `data` directory so you don't have to download data externally to run the notebooks.
