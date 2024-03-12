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


    def chi_sq(self, params: np.ndarray, anode_dist: np.ndarray) -> float:

        # compute model
        model = self.modulation_curve(self.theta, params)

        # get joint chi sq
        chi_sq = np.sum((anode_dist - model)**2 / anode_dist)

        return chi_sq
    
    
    def fit(self, anode_dist: np.ndarray) -> np.ndarray:
        """Fit modulation curve to the data and return the value of the fit parameters
        """
        
        # Check if fitting anode wise modulation or corrected modulation
        if anode_dist.ndim == 2:
            # Define initial guesses for each anode
            initial_params = np.zeros((anode_dist.shape[1], 3))
            initial_params[:, 0] = anode_dist.min(0)  # A
            initial_params[:, 1] = anode_dist.max(0) - anode_dist.min(0)  # B
        
            # Define initial guess for phase based on pure geometry
            anode_centre = np.arange(16.5, -16.6, -3)
            initial_params[:, 2] = np.tile(np.arctan2(anode_centre, 23)), 4 + np.repeat([0, -np.pi/2, -np.pi, -3*np.pi/2], 12)
            initial_params[initial_params[:, 2] < 0, 2] += 2*np.pi
        
            # define bounds (restrict phase in +/- 5 deg)
            bounds = Bounds(lb=np.hstack((np.zeros(48)[:, None], np.zeros(48)[:, None], initial_params[:, 2][:, None] - 0.035)).ravel(),
                            ub=np.hstack((np.repeat(np.inf, 48)[:, None], np.repeat(np.inf, 48)[:, None], initial_params[:, 2][:, None] + 0.035)).ravel())
        
        else:
            # Define initial guesses for each anode
            initial_params = np.zeros((3,))
            initial_params[0] = anode_dist.min()  # A
            initial_params[1] = anode_dist.max() - anode_dist.min()  # B
            initial_params[2] = np.arctan2(16.5, 23) # fix to anode 1
            
            # define bounds (restrict phase in +/- 5 deg)
            bounds = Bounds(lb=[0, 0, initial_params[2] - 2], ub=[initial_params[0], initial_params[1],
                                                                      initial_params[2] + 2])
            
        # fit
        opt_params = minimize(self.chi_sq, initial_params.ravel(), args=(anode_dist), method="Nelder-Mead", bounds=bounds).x

        # reshape if multi anode fit
        if anode_dist.ndim == 2:
            opt_params = opt_params.reshape(48, 3)

        return opt_params


    def apply_phase_shift(self, PA_in: np.array, phases: np.array, anode_dist: np.ndarray) -> np.array:
        phase_shifted_dist = np.zeros((anode_dist.size, 2))
        for anode in range(anode_dist.shape[1]):
            phase_shifted_dist[anode*360: (anode+1)*360, 0] = (PA_in - phases[anode] + 360) % 360
            phase_shifted_dist[anode*360: (anode+1)*360, 1] = anode_dist[:, anode]

        return phase_shifted_dist
