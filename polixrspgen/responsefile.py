import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table


class POLIXResponseFile:
    
    def __init__(self, rsptype: str) -> None:
        self.rsptype = rsptype
        
        self.polixrspHDU = fits.HDUList()
        self.polixrspHDU.append(fits.PrimaryHDU())
        
        # Common headers
        self.commonHDR = fits.header.Header()
        self.commonHDR['TELESCOP'] = ('XPoSat', 'Telescope (mission) name')
        self.commonHDR['INSTRUME'] = ('POLIX', 'Instrument name')
        
    def append_inebounds(self, Elow: np.array, Ehigh: np.array) -> None:

        INEBOUNDS = Table(data=[Elow, Ehigh], names=['ENERG_LO', 'ENERG_HI'], dtype=[np.float32, np.float32])
        
        # Append the table to fits
        self.polixrspHDU.append(fits.BinTableHDU(INEBOUNDS, name="INEBOUNDS"))
        
    
    def append_inpabounds(self, PA: np.array) -> None:
        
        INPAVALS = Table(data=[INPAVALS], names=['PA_IN'], dtype=[np.float32])
        
        # Append the table to fits
        self.polixrspHDU.append(fits.BinTableHDU(INPAVALS, name="INPAVALS"))
        
    
    def append_polmatrix(self, matrix: np.ndarray) -> None:
        
        # Append the image to fits
        self.polixrspHDU.append(fits.ImageHDU(matrix, name="POLMATRIX"))
        
    
    def write(self, outpath: Path) -> None:
        
        if self.rsptype == "pol":
            fname = outpath.joinpath("POLIX_polresponse.prsp")
            self.polixrspHDU.writeto(fname.as_posix(), overwrite=True)
