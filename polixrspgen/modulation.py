import numpy as np
from scipy.optimize import minimize, Bounds
from typing import Tuple

class Modulation:
    
    def __init__(self, PAin: np.array) -> None:
        
        self.theta = PAin

    def modulation_curve(self, phi: float or np.array, params: np.array or np.ndarray) -> float or np.array:
        """ Amplitude modulation function"""
        
        # convert to radians
        phi = np.deg2rad(phi)
               
        if params.size // 3 == 1:
            return params[0] + params[1]*np.cos(phi - params[2])**2
        else:
            # reshape params
            params = params.reshape((params.size // 3, 3))
            return params[:, 0] + params[:, 1]*np.cos(phi[:, None] - params[:, 2])**2


    def _chi_sq(self, params: np.ndarray, anode_dist: np.ndarray) -> float:

        # compute model
        model = self.modulation_curve(self.theta, params)

        # get joint chi sq
        chi_sq = np.sum((anode_dist - model)**2 / anode_dist)

        return chi_sq
    
    
    def fit(self, anode_dist: np.ndarray) -> np.ndarray:
        """Fit modulation curve to the data and return the value of the fit parameters
        """
 
        # Define initial guesses for each anode
        initial_params = np.zeros((anode_dist.shape[1], 3))
        initial_params[:, 0] = anode_dist.min(0)  # A
        initial_params[:, 1] = anode_dist.max(0) - anode_dist.min(0)  # B
        
        # Define initial guess for phase based on pure geometry
        anode_centre = np.arange(16.5, -16.6, -3)
        initial_params[:, 2] = np.tile(np.degrees(np.arctan2(anode_centre, 23)), 4) + np.repeat([0, -90, -180, -270], 12)
        initial_params[initial_params[:, 2] < 0, 2] += 360
        initial_params[:, 2] = np.deg2rad(initial_params[:, 2])
        
        # define bounds (restrict phase in +/- 5 deg)
        bounds = Bounds(lb=np.hstack((np.zeros(48)[:, None], np.zeros(48)[:, None], initial_params[:, 2][:, None] - 0.035)).ravel(),
                ub=np.hstack((np.repeat(np.inf, 48)[:, None], np.repeat(np.inf, 48)[:, None], initial_params[:, 2][:, None] + 0.035)).ravel())
        
        # fit
        opt_params = minimize(self._chi_sq, initial_params.ravel(), args=(anode_dist), method="Nelder-Mead", bounds=bounds).x
        opt_params = opt_params.reshape(48, 3)

        return opt_params


    def get_anode_phases(self, anode_dist: np.ndarray) -> np.ndarray:
        
        nE = anode_dist.shape[0]
        anode_phases = np.zeros((nE, 48))
        for i in range(nE):
            anode_phases[i, :] = self.fit(anode_dist[i, :, :])[:, 2]
            
        return anode_phases