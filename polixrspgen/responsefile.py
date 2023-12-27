import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table


class POLIXResponseFile:
    
    def __init__(self, rsptype: str, nchans=None) -> None:
        self.rsptype = rsptype
        self.nchans = nchans
        
        self.polixrspHDU = fits.HDUList()
        self.polixrspHDU.append(fits.PrimaryHDU())
        
        # Common headers
        self.commonHDR = fits.header.Header()
        self.commonHDR['TELESCOP'] = ('XPoSat', 'Telescope (mission) name')
        self.commonHDR['INSTRUME'] = ('POLIX', 'Instrument name')
        self.commonHDR['FILTER'] = ('NONE ', 'Instrument filter')
        
        if rsptype == "spectral":
            self.commonHDR['CHANTYPE'] = ('PHA', 'Type of channels (PHA, PI etc)')
            self.commonHDR['DETCHANS'] = (self.nchans, 'Total number of detector PHA channels')
        
    def append_inebounds(self, Elow: np.array, Ehigh: np.array) -> None:

        INEBOUNDS = Table(data=[Elow, Ehigh], names=['ENERG_LO', 'ENERG_HI'], dtype=[np.float16, np.float16],
                          units=['keV', 'keV'])
        
        # Append the table to fits
        self.polixrspHDU.append(fits.BinTableHDU(INEBOUNDS, name="INEBOUNDS"))
        HDR =self.polixrspHDU[-1].header
        HDR.extend(self.commonHDR)
        
    
    def append_inpabounds(self, PA: np.array) -> None:
        
        INPAVALS = Table(data=[PA], names=['PA_IN'], dtype=[np.float16], units=['deg'])
        
        # Append the table to fits
        self.polixrspHDU.append(fits.BinTableHDU(INPAVALS, name="INPAVALS"))
        HDR =self.polixrspHDU[-1].header
        HDR.extend(self.commonHDR)
        
    
    def append_polmatrix(self, matrix: np.ndarray) -> None:
        
        # Append the image to fits
        self.polixrspHDU.append(fits.ImageHDU(matrix, name="POLMATRIX"))
        HDR = self.polixrspHDU[-1].header
        HDR['EMIN'] = (8., 'Min. det energy used (in keV)')
        HDR['EMAX'] = (50., 'Min. det energy used (in KeV)')
        HDR['UNIT'] = ('counts', 'Unit of matrix value')
        HDR.extend(self.commonHDR)
        
    
    def append_ebounds(self, Emin: np.array, Emax: np.array) -> None:
        
        CHAN = np.arange(0, self.nchans)
        EBOUNDS = Table(data=[CHAN, Emin, Emax], names=['CHANNEL', 'E_MIN', 'E_MAX'],
                        dtype=[np.int16, np.float16, np.float16], units=['', 'keV', 'keV'])
        
        # Append the channel ebounds extension
        self.polixrspHDU.append(fits.BinTableHDU(EBOUNDS, name="EBOUNDS"))
        HDR = self.polixrspHDU[-1].header
        HDR['TLMIN1'] = (0, 'First legal channel number')
        HDR['TLMAX2'] = (self.nchans, 'Highest legal channel number')
        HDR.extend(self.commonHDR)
        HDR['HDUCLASS'] = ('OGIP ', 'Organisation devising file format')
        HDR['HDUCLAS1'] = ('RESPONSE', 'File relates to response of instrument')
        HDR['HDUCLAS2'] = ('EBOUNDS', 'Extension contains a response Ebounds')
        HDR['HDUVERS1'] = ('1.1.0 ', 'Version of file format')
   
    
    def append_rspmatrix(self, Elow: np.array, Ehigh:np.array, matrix: np.ndarray) -> None:
        
        NGRP = np.ones((Elow.size))
        FCHAN = np.zeros((Elow.size,))
        NCHAN = np.repeat(self.nchans, Elow.size)
        colnames = ['ENERG_LO', 'ENERG_HI', 'N_GRP', 'F_CHAN', 'N_CHAN', 'MATRIX']
        specrsp_matrix = Table(data=[Elow, Ehigh, NGRP, FCHAN, NCHAN, matrix], names=colnames,
                               dtype=[np.float32, np.float32, np.int16, np.int16, np.int16, np.float32],
                               units=['keV', 'keV', '', '', '', 'cm**2'])
        
        # Append the rsp matrix extension
        self.polixrspHDU.append(fits.BinTableHDU(specrsp_matrix, name="SPECRESP MATRIX"))
        HDR = self.polixrspHDU[-1].header
        HDR['TLMIN4'] = (0, 'First legal channel number')
        HDR['TLMAX4'] = (self.nchans, 'Highest legal channel number')
        HDR.extend(self.commonHDR)
        HDR['LO_THRES'] = (1.00E-07, 'Lower probability density threshold for matrix')
        HDR['HDUCLASS'] = ('OGIP ', 'Organisation devising file format')
        HDR['HDUCLAS1'] = ('RESPONSE', 'File relates to response of instrument')
        HDR['HDUCLAS2'] = ('RSP_MATRIX', 'Extension contains a response matrix')
        HDR['HDUCLAS3'] = ('FULL', 'REDIST/DETECTOR/FULL')
        HDR['HDUVERS1'] = ('1.1.0 ', 'Version of file format')
        HDR['NUMGRP'] = (self.nchans, 'Total number of channel subsets')
        HDR['NELT'] = (self.nchans*Elow.size, 'Total number response elements')
 
        
    def append_arf(self, Elow: np.array, Ehigh: np.array, eff_area: np.array) -> None:
        
        arf = Table(data=[Elow, Ehigh, eff_area], names=['ENERG_LO', 'ENERG_HI', 'SPECRESP'],
                    dtype=[np.float32, np.float32, np.float32], units=['keV', 'keV', 'cm**2'])
        
        # Append the arf extension
        self.polixrspHDU.append(fits.BinTableHDU(arf, name="SPECRESP"))
        HDR = self.polixrspHDU[-1].header
        HDR.extend(self.commonHDR)
        HDR['HDUCLASS'] = ('OGIP ', 'Organisation devising file format')
        HDR['HDUCLAS1'] = ('RESPONSE', 'File relates to response of instrument')
        HDR['HDUCLAS2'] = ('SPECRESP', 'effective area data is stored')
        HDR['HDUVERS'] = ('1.1.0 ', 'Version of file format')
    
    
    def write(self, outpath: Path) -> None:
        
        if self.rsptype == "polarisation":
            fname = outpath.joinpath("POLIX_response.prsp")
            self.polixrspHDU.writeto(fname.as_posix(), overwrite=True)
        
        if self.rsptype == "area":
            fname = outpath.joinpath("POLIX_response.arf")
            self.polixrspHDU.writeto(fname.as_posix(), overwrite=True)
            
        if self.rsptype == "spectral":
            fname = outpath.joinpath("POLIX_response.rsp")
            self.polixrspHDU.writeto(fname.as_posix(), overwrite=True)
