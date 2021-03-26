# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:41:47 2021

@author: Nils
"""

import gpytorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.decomposition import PCA
import pickle
from scipy.linalg import svd, eig
import h5py

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel)

import gpytorch
from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior

import GPy
from emukit.model_wrappers import GPyModelWrapper

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ASM:
    
    def __init__(self, *params):
        
        self.gp_framework = "gpy"
    
    def load_dvs(self):
        pass

    def load_h5_data(self, fn, verbose=True):
        CD = []
        CL = []
        CMz = []
        RMS_Rho = []
        dsn_names = []

        d_CD = []

        #fn = "T007.h5"
        with h5py.File(fn, "r") as db:
            i = 0
            for dsn_name, dsn in db['DESIGNS'].items():
                if "DIRECT/history_direct.csv" in dsn and "ADJOINT_DRAG/of_grad_cd.csv/GRADIENT" in dsn:
                    if verbose:
                        print (dsn_name, dsn)
                    CD.append(dsn["DIRECT/history_direct.csv/CD"][-1])
                    CL.append(dsn["DIRECT/history_direct.csv/CL"][-1])
                    CMz.append(dsn["DIRECT/history_direct.csv/CMz"][-1])
                    RMS_Rho.append(dsn["DIRECT/history_direct.csv/rms[Rho]"][-1])
                    
                    d_CD.append(np.array(dsn["ADJOINT_DRAG/of_grad_cd.csv/GRADIENT"]))
                    
                    dsn_names.append(dsn_name)
                    i += 1
                else:
                    if verbose:
                        print(f"could not parse {dsn_name}")
        
        return dsn_names, CL, CD, CMz, RMS_Rho, d_CD

    def calculate_subspace(self, dY, energy_criterium=0.95, rank=None, method="eig", center_gradient=False):

        dY = np.copy(dY)

        n_samples, D = dY.shape[0], dY.shape[1]

        if center_gradient:
            dY_mean = np.mean(dY, axis=0)
            dY = dY - dY_mean

        C = np.dot(dY.T, dY) / n_samples

        # C = np.zeros((D,D))
        # for i in range(n_samples):
        #     C += np.outer(dY[i], dY[i])
        # C = C / n_samples

        if method == "eig":
            l, W = linalg.eig(C)
            l = l.astype("double")
            self.singular_values_ = l
            explained_variance_ = (l ** 2) / (n_samples - 1)
            total_var = explained_variance_.sum()
            self.explained_variance_ratio_ = explained_variance_ / total_var
            components_ = W.T

        elif method == "svd":
            U, S, Vt = linalg.svd(C)
            
            explained_variance_ = (S ** 2) / (n_samples - 1)
            total_var = explained_variance_.sum()
            self.explained_variance_ratio_ = explained_variance_ / total_var
            self.singular_values_ = S.copy()  # Store the singular values.
            #pass
            W[:,1] = Vt
        
        if rank:
            self.components_ = components_[:rank]
        
        elif energy_criterium:
            n_components = np.searchsorted(self.explained_variance_ratio_, energy_criterium, side='right') + 1
            self.components_ = components_[:n_components]
        


    def plot_eigenmodes(self):

        print("plot")
        plt.plot(range(1,39), self.singular_values_.cumsum() / self.singular_values_.sum(), "-s")
        plt.hlines([0.99], 0, 38, ls="--")
        plt.gca().set_xlabel("Eigenmodes")
        plt.gca().set_ylabel("Cum. Fraction of Total EV")
    
    def fit_gp(self, X, Y, *params):

        input_dim = X.shape[1]
        
        if "framework" in params:
            self.gp_framework = params["gp_framework"]
        else:
            self.gp_framework = "gpy"
        
        if "length_scale" in params:
            self.length_scale = params["length_scale"]
        else:
            self.length_scale = 0.001
        
        if self.gp_framework == "sklearn":

            #kernel = Matern(length_scale=ls, nu=1.5) +  WhiteKernel(noise_level=1e-05, noise_level_bounds=(1e-09, 1e-04))
            #kernel = RBF(length_scale=ls, length_scale_bounds=(1e-09, 1e04)) + WhiteKernel(noise_level=1e-05, noise_level_bounds=None)
            #self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=False)
            #self.gp.fit(x_train, y_train)
            pass
        
        elif self.gp_framework == "emukit" or self.gp_framework == "gpy":

            print("gpy")

            kernel = GPy.kern.src.rbf.RBF(input_dim, ARD=True)
            self.gpy_model = GPy.models.GPRegression(X, Y, kernel=kernel)
            self.gpy_model.optimize(optimizer=None, messages=True,max_iters=1000)

    def predict(self, X):
        pass
        
    def plot_result(self):
        pass