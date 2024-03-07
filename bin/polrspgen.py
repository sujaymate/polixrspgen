#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
from polixrspgen.simulation import Simulation
from polixrspgen.responsefile import POLIXResponseFile
from polixrspgen.modulation import Modulation
    
parser = argparse.ArgumentParser(description='Estimate modulation factor for POLIX')
parser.add_argument("simdatadir", type=str, help='Simulation data path')
parser.add_argument("outdir", type=str, help='Simulation data path')
args = parser.parse_args()


# Data paths
simdatapath = Path(args.simdatadir)
outdatapath = Path(args.outdir)

# Read simulated data and create response
simulation = Simulation(simdatapath)

# Pol response
anode_dist, Ein, PAin = simulation.read_polrsp_data()
modulation = Modulation(PAin)
anode_phases = np.rad2deg(modulation.fit(anode_dist.sum(0))[:, 2])

# Spectral response
Ein, eff_area = simulation.read_specrsp_data("area")
Ein, Ebins_out, rspmatrix = simulation.read_specrsp_data("spectral")

# Prepare to write data
deltaE = np.diff(Ein)[0]
Elow = Ein - deltaE / 2
Ehigh = Ein + deltaE / 2

anode_dist = anode_dist.astype(np.uint16)

# Open polarisation response file
prspfile = POLIXResponseFile("polarisation")
prspfile.append_inebounds(Elow, Ehigh)
prspfile.append_inpabounds(PAin)
prspfile.append_polmatrix(anode_dist)
prspfile.append_anode_phases(anode_phases)
prspfile.write(outdatapath)

# Open area response file

arffile = POLIXResponseFile("area")
arffile.append_arf(Elow, Ehigh, eff_area)
arffile.write(outdatapath)

# Open spectral response file

rspfile = POLIXResponseFile("spectral", 8192)
rspfile.append_ebounds(Ebins_out[:-1], Ebins_out[1:])
rspfile.append_rspmatrix(Elow, Ehigh, rspmatrix)
rspfile.write(outdatapath)