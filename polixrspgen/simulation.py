from pathlib import Path
from typing import Tuple

import numpy as np
from astropy.io import fits


class Simulation:
    
    def __init__(self, simdatapath: Path) -> None:
        
        # Energy grid in sim
        self.simEmin = 8.
        self.simEmax = 50.
        self.dE = 1.
        self.Ein = np.arange(self.simEmin, self.simEmax + self.dE/2, self.dE)
        self.nE = self.Ein.size

        # PA grid in sim
        self.PAmin = 0.
        self.PAmax = 359.
        self.dPA = 1.
        self.PAin = np.arange(self.PAmin, self.PAmax + self.dPA/2, self.dPA)
        self.nPA = self.PAin.size
        
        # cal params
        self.eres = 23.
        self.nchans = 8192
        
        self.anode_dist = np.zeros((self.nE, self.nPA, 48))
        
        self.simdatapath = simdatapath
           
    def read_polrsp_data(self) -> Tuple[np.ndarray, np.array, np.array]:
        
        for i, Ein in enumerate(self.Ein):
            for j, PA in enumerate(self.PAin):
                simevtfname = self.simdatapath.joinpath(f"XPoSatEvents_MonoE_{Ein:02.0f}_PA_{PA:03.0f}.fits.gz")
                print(f"Processing {simevtfname.stem:s}")
                
                eventHDU = fits.open(simevtfname)
                self.anode_dist[i, j, :] = self._get_pol_response(eventHDU, Ein)

        return self.anode_dist, self.Ein, self.PAin
    
    
    def read_specrsp_data(self, rsptype: str) -> Tuple[np.array, np.array] or Tuple[np.array, np.array, np.ndarray]:

        if rsptype == "area":
            rspdata = np.zeros((self.nE))
        else:
            rspdata = np.zeros((self.nE, self.nchans))

        for i, Ein in enumerate(self.Ein):
            simevtfname = self.simdatapath.joinpath(f"XPoSatEvents_MonoE_{Ein:02.0f}_PA_999.fits.gz")
            print(f"Processing {simevtfname.stem:s}")
            eventHDU = fits.open(simevtfname)
            events = eventHDU[1].data
            in_flux = eventHDU[1].header['PRIMFLUX']
            
            events = self._apply_energy_resolution(events)

            # Discard residual anti coincidence events
            events = events[~(events['anodeID']==0)]

            # Only select single events
            events = events[events['mult']==1]

            # create response according to the type
            if rsptype == "area":
                events = events[(events['energy'] >= 8.) & (events['energy'] <= 50.)]
                rspdata[i] = events.size / in_flux  # area in cm2
            
            else:
                Ebins_out = np.linspace(3-0.008/2, (self.nchans + 1)*.008, self.nchans + 1)
                rspdata[i, :] = np.histogram(events['energy'], bins=Ebins_out)[0]
            
        if rsptype == "area":
            return self.Ein, rspdata
        else:
            return self.Ein, Ebins_out, rspdata
    
    
    def _apply_energy_resolution(self, events: fits.fitsrec.FITS_rec) -> fits.fitsrec.FITS_rec:

        sigma = events['energy'] * (self.eres / 100) / 2.355
        events['energy'] = np.random.normal(events['energy'], sigma)
        
        return events
    
    
    def _get_spectral_response(self, events: fits.fitsrec.FITS_rec) -> np.ndarray:
        pass


    def _get_area_response(self, events: fits.fitsrec.FITS_rec) -> np.array:
        pass

    
    def _get_pol_response(self, eventHDU: fits.hdu.hdulist.HDUList, Ein: float) -> np.array:
        
        events = eventHDU[1].data
        events = self._apply_energy_resolution(events)
        
        # Discard residual anti coincidence events
        events = events[~(events['anodeID']==0)]
        
        # Only select single events
        events = events[events['mult']==1]
        
        # Only select events in +/- sigma keV range around the central energy
        sigma = Ein * (self.eres / 100) / 2.355
        events = events[(events['energy'] >= (Ein - sigma)) & (events['energy'] <= (Ein + sigma))]
        
        # Get final anode counts
        anodeID_global = (events['detID'] - 1)*12 + events['anodeID']
        anode, counts = np.unique(anodeID_global, return_counts=True)
        
        return counts