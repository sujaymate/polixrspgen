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

        INEBOUNDS = Table(data=[Elow, Ehigh], names=['ENERG_LO', 'ENERG_HI'], dtype=[np.float16, np.float16])
        
        # Append the table to fits
        self.polixrspHDU.append(fits.BinTableHDU(INEBOUNDS, name="INEBOUNDS"))
        HDR =self.polixrspHDU[-1].header
        HDR['XTENSION'] = ('BINTABLE', 'binary table extension')
        HDR['NAXIS2'] = (Elow.size, 'number of rows in table')
        HDR['PCOUNT'] = (0, 'size of special data area')
        HDR['GCOUNT'] = (1, 'one data group (required keyword)')
        HDR['TFIELDS'] = (2, 'number of fields in each row')
        HDR['TTYPE1'] = ('ENERG_LO', 'label for field 1')
        HDR['TFORM1'] = ('1E ', 'data format of field: 4-byte REAL')
        HDR['TUNIT1'] = ('keV ', 'physical unit of field')
        HDR['TTYPE2'] = ('ENERG_HI', 'label for field 2')
        HDR['TFORM2'] = ('1E ', 'data format of field: 4-byte REAL')
        HDR['TUNIT2'] = ('keV ', 'physical unit of field')
        HDR['EXTNAME'] = ('INEBOUNDS', 'Name of the binary table extension')
        HDR.extend(self.commonHDR)
        
    
    def append_inpabounds(self, PA: np.array) -> None:
        
        INPAVALS = Table(data=[PA], names=['PA_IN'], dtype=[np.float16])
        
        # Append the table to fits
        self.polixrspHDU.append(fits.BinTableHDU(INPAVALS, name="INPAVALS"))
        HDR =self.polixrspHDU[-1].header
        HDR['XTENSION'] = ('BINTABLE', 'binary table extension')
        HDR['NAXIS2'] = (PA.size, 'number of rows in table')
        HDR['PCOUNT'] = (0, 'size of special data area')
        HDR['GCOUNT'] = (1, 'one data group (required keyword)')
        HDR['TFIELDS'] = (1, 'number of fields in each row')
        HDR['TTYPE1'] = ('PA_IN', 'label for field 1')
        HDR['TFORM1'] = ('1E ', 'data format of field: 4-byte REAL')
        HDR['TUNIT1'] = ('deg ', 'physical unit of field')
        HDR['EXTNAME'] = ('INPAVALS', 'Name of the binary table extension')
        HDR.extend(self.commonHDR)
        
    
    def append_polmatrix(self, matrix: np.ndarray) -> None:
        
        # Append the image to fits
        self.polixrspHDU.append(fits.ImageHDU(matrix, name="POLMATRIX"))
        HDR = self.polixrspHDU[-1].header
        HDR['XTENSION'] = ('IMAGE', 'image extension')
        HDR['EXTNAME'] = ('POLMATRIX', 'Name of the binary table extension')
        HDR['EMIN'] = (8., 'Min. det energy used (in keV)')
        HDR['EMAX'] = (50., 'Min. det energy used (in KeV)')
        HDR['UNIT'] = ('counts', 'Unit of matrix value')
        HDR.extend(self.commonHDR)
        
    
    def append_ebounds(self, Emin: np.array, Emax: np.array, Nchans: int) -> None:
        
        CHAN = np.arange(0, Nchans)
        EBOUNDS = Table(data=[CHAN, Emin, Emax], names=['CHANNEL', 'E_MIN', 'E_MAX'],
                        dtype=[np.int16, np.float16, np.float16])
        
        # Append the channel ebounds extension
        self.polixrspHDU.append(fits.BinTableHDU(EBOUNDS, name="EBOUNDS"))
        HDR = self.polixrspHDU[-1].header
        
    
    def append_rspmatrix(self, Elow: np.array, Ehigh:np.array, matrix: np.ndarray, Nchans: int) -> None:
        
        NGRP = np.ones((Elow.size))
        FCHAN = np.zeros((Elow.size,))
        NCHAN = np.repeat(Nchans, Elow.size)
        colnames = ['ENERG_LO', 'ENERG_HI', 'N_GRP', 'F_CHAN', 'N_CHAN', 'MATRIX']
        specrsp_matrix = Table(data=[Elow, Ehigh, NGRP, FCHAN, NCHAN, matrix], names=colnames,
                               dtype=[np.float32, np.float32, np.int16, np.int16, np.int16, np.float32])
        
        # Append the rsp matrix extension
        self.polixrspHDU.append(fits.BinTableHDU(specrsp_matrix, name="SPECRESP MATRIX"))
        HDR = self.polixrspHDU[-1].header
        
        
    def append_arf(self, Elow: np.array, Ehigh: np.array, eff_area: np.array) -> None:
        
        arf = Table(data=[Elow, Ehigh, eff_area], names=['ENERG_LO', 'ENERG_HI', 'SPECRESP'],
                    dtype=[np.float32, np.float32, np.float32])
        
        # Append the arf extension
        self.polixrspHDU.append(fits.BinTableHDU(arf, name="SPECRESP"))
        HDR = self.polixrspHDU[-1].header
    
    
    def write(self, outpath: Path) -> None:
        
        if self.rsptype == "polarisation":
            fname = outpath.joinpath("POLIX_polresponse.prsp")
            self.polixrspHDU.writeto(fname.as_posix(), overwrite=True)
        
        if self.rsptype == "area":
            fname = outpath.joinpath("POLIX_polresponse.arf")
            self.polixrspHDU.writeto(fname.as_posix(), overwrite=True)
            
        if self.rsptype == "spectral":
            fname = outpath.joinpath("POLIX_polresponse.rsp")
            self.polixrspHDU.writeto(fname.as_posix(), overwrite=True)
