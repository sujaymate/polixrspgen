#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from polixrspgen.modulation import Modulation


def plaw(E: np.array, norm: float, alpha: float) -> np.array:
    return norm*E**alpha


parser = argparse.ArgumentParser(description='Estimate modulation factor for POLIX')
parser.add_argument("prspfile", type=str, help='Polarisation response file')
parser.add_argument("rspfile", type=str, help='Spectral response file')
parser.add_argument("outpath", type=str, help='Path to store output')
parser.add_argument("--norm", type=float, default=10, help='Norm value for the source at 1 keV in ph/s/keV/cm2')
parser.add_argument("--alpha", type=float, default=-2.1, help='Powerlaw index')
args = parser.parse_args()

# read pol response files
polrspHDU = fits.open(Path(args.prspfile))
Ebins = np.append((polrspHDU[1].data['ENERG_LO']), polrspHDU[1].data['ENERG_HI'][-1])
Ebins_centre = (polrspHDU[1].data['ENERG_HI'] + polrspHDU[1].data['ENERG_LO']) / 2
PA = polrspHDU[2].data['PA_IN']
anode_dist = polrspHDU[3].data
phases = polrspHDU[4].data['PHASE']

# get relative phases wrt to anode 1
phases = phases - phases[0]

# read spectral response
specrspHDU = fits.open(Path(args.rspfile))
Ebin_out = (specrspHDU[1].data['E_min'] + specrspHDU[1].data['E_Max']) / 2
matrix = specrspHDU[2].data['MATRIX']

# Convolve spectral response to get detected spectra
in_spec = plaw(Ebins_centre, args.norm, args.alpha)
spec = matrix.T.dot(in_spec)

# rebin data in 8 - 50 keV
n_8_50, bins = np.histogram(Ebin_out, bins=Ebins, weights=spec)

# Phase shift and create flat pol-response
modulation = Modulation(PA)
ps_pol_rsp = np.zeros((Ebins_centre.size, 360))
for i, E in enumerate(Ebins_centre):
    n_total = anode_dist[i].sum()
    anode_dist_ps = modulation.apply_phase_shift(PA, phases, anode_dist[i])
    ps_pol_rsp[i, :], PAbins = np.histogram(anode_dist_ps[:, 0], bins=np.arange(0, 360.5, 1),
                                            weights=anode_dist_ps[:, 1])
    ps_pol_rsp[i, :] /= n_total

# Get final modulation curve for 8 - 30 and 8 - 50
src_modulation = ps_pol_rsp * n_8_50[:, None]
sel = Ebins_centre <= 30
src_modulation_8_50 = src_modulation.sum(0)
src_modulation_8_30 = src_modulation[sel].sum(0)

# fit to get mu
opt_params_8_50 = modulation.fit(src_modulation_8_50)
opt_params_8_30 = modulation.fit(src_modulation_8_30)

fit_8_50_keV = modulation.modulation_curve(PA, opt_params_8_50)
fit_8_30_keV = modulation.modulation_curve(PA, opt_params_8_30)

# get mu
mu_8_50 = opt_params_8_50[1] / (opt_params_8_50[0]*2 + opt_params_8_50[1])
mu_8_30 = opt_params_8_30[1] / (opt_params_8_30[0]*2 + opt_params_8_30[1])

# plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(PA, src_modulation_8_30, 'o', ms=4, label=f"8 - 30 keV; $\mu$ = {mu_8_30:05.3f}")
ax.plot(PA, fit_8_30_keV, 'k')
ax.plot(PA, src_modulation_8_50, 'o', ms=4, label=f"8 - 50 keV; $\mu$ = {mu_8_50:05.3f}")
ax.plot(PA, fit_8_50_keV, 'k')
ax.legend(fontsize=12)
ax.set_xlabel("Azimuthal angle (deg)", fontsize=15)
ax.set_ylabel("Counts/s", fontsize=15)
fig.tight_layout()
fig.savefig(Path(args.outpath).joinpath("Expected_Source_Modulation.png"), dpi=150)
plt.show()