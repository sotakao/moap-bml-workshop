# Gaussian Process Tutorials for UCL-MOAP Workshop

This repo contains the tutorials that I made in companion to my presentation at the UCL-MOAP workshop on "Bayesian Machine Learning for Weather and Climate". In particular, you can find implementations of all the experiments shown in the presentation and further details on various techniques that I didn't have time to present at the workshop.

The tutorial is divided into 5 parts, which cover the following topics.
1. Basics of Gaussian process regression including how to set up a model, how to condition on data and how to optimise for the hyperparameters using the log marginal likelihood.
2. Sparse GPs, in particular Titsisas' inducing point method. Includes information on the mathematical derivation, comparisons with a vanilla Gaussian process and training.
3. Gaussian process inference for timeseries data using Kalman filtering or smoothing techniques.
4. Global temperature interpolation using Matérn GPs defined on the sphere.
5. Global wind speed interpolation using projected kernels.

## Requirements
To run the first three notebooks, please install the packages in the `requirements.txt` file by running
```
pip install -r requirements.txt
```
The codes in the fourth and fifth tutorials use our `riemannianvectorgp` package (can be found [here](https://github.com/MJHutchinson/ExtrinsicGaugeIndependentVectorGPs)), which is not on pip so you have to clone this into your repository and install the requirements therein. To plot the results, I used the `cartopy` package, which is easiest to install using conda.

## Data
All the datasets used in the tutorials are saved in the `data` directory.
