import numpy as np
from scipy.optimize import minimize, curve_fit


class Modulation:
    
    def __init__(self) -> None:
        pass

    def modulation_curve(self, phi: float or np.array, A: float or np.array, B: float or np.array, phase: float or np.array):
        """ Amplitude modulation function"""
        return A*np.cos(2*(phi + phase + np.pi/2)) + B


    def joint_chi_sq(self, params: np.ndarray, phi: np.array, counts: np.ndarray):
        # prepare arrays
        params = np.repeat(params.reshape(counts.shape[0], 3), phi.size, 0)
        thetas = np.tile(phi, counts.shape[0])

        # compute model
        model = self.modulation_curve(thetas, params[:, 0], params[:, 1], params[:, 2])
        model = model.reshape(counts.shape)

        # get joint chi sq
        chi_sq = np.sum((counts - model)**2 / counts)

        return chi_sq