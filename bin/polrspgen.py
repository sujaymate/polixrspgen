#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
from polixrspgen.simulation import Simulation
from polixrspgen.responsefile import POLIXResponseFile
    
parser = argparse.ArgumentParser(description='Estimate modulation factor for POLIX')
parser.add_argument("simdatadir", type=str, help='Simulation data path')
parser.add_argument("outdir", type=str, help='Simulation data path')
args = parser.parse_args()


# Data paths
simdatapath = Path(args.simdatadir)
outdatapath = Path(args.outdir)

# Read simulated data
simulation = Simulation(simdatapath)
anode_dist, Ein, PAin = simulation.read_polrsp_data()

# Prepare to write data
deltaE = np.diff(Ein)[0]
Ein_min = Ein - deltaE / 2
Ein_max = Ein + deltaE / 2

anode_dist = anode_dist.astype(np.uint16)

# Open response file
rspfile = POLIXResponseFile("polarisation")
rspfile.append_inebounds(Ein_min, Ein_max)
rspfile.append_inpabounds(PAin)
rspfile.append_polmatrix(anode_dist)
rspfile.write(outdatapath)