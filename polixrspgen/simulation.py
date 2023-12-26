from pathlib import Path
from typing import Tuple

import numpy as np
from astropy.io import fits


class Simulation:
    
    def __init__(self, simdatapath: Path) -> None:
        
        self.nE = 43
        self.nPA = 180
        self.simdatapath = simdatapath
        self.eres = 23.
        self.anode_dist = np.zeros((self.nE, self.nPA, 48))
        self.Ein = np.zeros((self.nE,))
        self.PAin = np.zeros((self.nPA,))
    
    def read_polrsp_data(self) -> Tuple[np.ndarray, np.array, np.array]:
        
        for i, Ein in enumerate(np.arange(8, 50.1, 1)):
            for j, PA in enumerate(np.arange(0, 180, 1)):
                simevtfname = self.simdatapath.joinpath(f"XPoSatEvents_MonoE_{Ein:02.0f}_PA_{PA:03.0f}.fits.gz")
                print(f"Processing {simevtfname.stem:s}")
                
                eventHDU = fits.open(simevtfname)
                self.anode_dist[i, j, :] = self._get_anode_counts(eventHDU, Ein)
                if self.PAin[j] == 0:
                    self.PAin[j] = PA
            
            self.Ein[i] = Ein
                
        return self.anode_dist, self.Ein, self.PAin
    
    
    def read_specrsp_data():
        pass
    
    
    def _apply_energy_response(self, events: fits.fitsrec.FITS_rec) -> fits.fitsrec.FITS_rec:

        sigma = events['energy'] * (self.eres / 100) / 2.355
        events['energy'] = np.random.normal(events['energy'], sigma)
        
        return events
    
    
    def _get_anode_counts(self, eventHDU: fits.hdu.hdulist.HDUList, Ein: float) -> np.array:
        
        events = eventHDU[1].data
        events = self._apply_energy_response(events)
        
        # Discard residual anti coincidence events
        events = events[~(events['anodeID']==0)]
        
        # Only select single events
        events = events[events['mult']==1]
        
        # Only select events in 1 keV range around the central energy
        events = events[(events['energy'] >= (Ein - 0.5)) & (events['energy'] >= (Ein + 0.5))]
        
        # Get final anode counts
        anodeID_global = (events['detID'] - 1)*12 + events['anodeID']
        anode, counts = np.unique(anodeID_global, return_counts=True)
        
        return counts