import numpy as np
from scipy.optimize import minimize, Bounds
from typing import Tuple

class Modulation:
    
    def __init__(self, PAin: np.array) -> None:
        
        self.theta = np.deg2rad(PAin)

    def modulation_curve(self, phi: float or np.array, params: np.array or np.ndarray) -> float or np.array:
        """ Amplitude modulation function"""
        
        if params.size // 3 == 1:
            return params[0]*np.cos(2*(phi - params[2])) + params[1]
        else:
            # reshape params
            params = params.reshape((params.size // 3, 3))
            return params[:, 0]*np.cos(2*(phi[:, None] - params[:, 2])) + params[:, 1]


    def _chi_sq(self, params: np.ndarray, anode_dist: np.ndarray) -> float:

        # compute model
        model = self.modulation_curve(self.theta, params)

        # get joint chi sq
        chi_sq = np.sum((anode_dist - model)**2 / anode_dist)

        return chi_sq
    
    
    def fit(self, anode_dist: np.ndarray) -> np.ndarray:
        """Fit modulation curve to the data and return the value of the fit parameters
        """
 
        initial_params = np.zeros((anode_dist.shape[1], 3))
        initial_params[:, 0] = (anode_dist.max(0) + anode_dist.min(0))/2  # A
        initial_params[:, 1] = (anode_dist.max(0) - anode_dist.min(0))/2  # B
        initial_params[:, 2] = np.linspace(0, 2*np.pi, 48)  # Anode phase
        
        bnds =  ((0, 2*initial_params[:, 0].max()), (0, 2*initial_params[:, 1].max()), (0, 2*np.pi))*48

        opt_params = minimize(self._chi_sq, initial_params.ravel(), args=(anode_dist), method="Powell", bounds=bnds).x
        opt_params = opt_params.reshape(48, 3)

        return opt_params


    def get_anode_phases(self, anode_dist: np.ndarray) -> np.ndarray:
        
        nE = anode_dist.shape[0]
        anode_phases = np.zeros((nE, 48))
        for i in range(nE):
            anode_phases[i, :] = np.rad2deg(self.fit(anode_dist[i, :, :])[:, 2])
            
        return anode_phases