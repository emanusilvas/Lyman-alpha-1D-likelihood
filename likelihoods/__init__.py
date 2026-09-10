################################################################
# DESI DR1 Lyman-alpha 1D likelihood
# Based on arXiv:2601.21432
# @moduleauthor:: Emanuelly Silva <emanuelly.santos@ufrgs.br>
#
################################################################

import os
import numpy as np
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
import scipy.constants as conts

C_KMS = 299792.458  # speed of light in km/s


def fit_polynomial(xmin, xmax, x, y, deg=2):
    x_fit = (x > xmin) & (x < xmax)
    poly = np.polyfit(np.log(x[x_fit]), np.log(y[x_fit]), deg=deg)
    return np.poly1d(poly)


class LyAlpha(Likelihood):
    def __init__(self, path, data, command_line):
        Likelihood.__init__(self, path, data, command_line)

        if not hasattr(self, 'kp'):
            self.kp = 0.009
        if not hasattr(self, 'zp'):
            self.zp = 3.0
        if not hasattr(self, 'deg'):
            self.deg = 2
        if not hasattr(self, 'fit_min'):
            self.fit_min = 0.5
        if not hasattr(self, 'fit_max'):
            self.fit_max = 2.0

        file_path = os.path.join(self.data_directory, self.file)
        loaded_data = np.loadtxt(file_path, ndmin=2)

        self.delta2_L = loaded_data[:, 0]
        self.n_L = loaded_data[:, 1]
        self.error_delta2_L = loaded_data[:, 2]
        self.error_n_L = loaded_data[:, 3]
        self.rho = loaded_data[:, 4]

        self.num_points = len(self.delta2_L)

    def Pk(self, cosmo):
    
        kstar_kms = self.kp
        zstar = self.zp

        k_kms = np.logspace(
            np.log10(self.fit_min * kstar_kms),
            np.log10(self.fit_max * kstar_kms),
            100,
        )

        H_z = cosmo.Hubble(zstar) * C_KMS
        dvdX = H_z / (1.0 + zstar)

        k_Mpc = k_kms * dvdX  # 1/Mpc

        P_Mpc = np.array([cosmo.pk_lin(k, zstar) for k in k_Mpc]) # (Mpc^3)

        P_kms = P_Mpc * dvdX**3

        P_fit = fit_polynomial(self.fit_min, self.fit_max, k_kms / kstar_kms, P_kms, deg=self.deg)

        Delta2_L = np.exp(P_fit[0]) * kstar_kms**3 / (2.0 * np.pi**2)
        n_L = P_fit[1]

        return Delta2_L, n_L

    def loglkl(self, cosmo, data):

        teo_delta2_L, teo_n_L = self.Pk(cosmo)

        delta_x = (teo_delta2_L - self.delta2_L) / self.error_delta2_L
        delta_y = (teo_n_L - self.n_L) / self.error_n_L

        chi2_terms = (delta_x**2 - 2 * self.rho * delta_x * delta_y + delta_y**2) / (1 - self.rho**2)
        chi2 = np.sum(chi2_terms)

        return -0.5 * chi2
