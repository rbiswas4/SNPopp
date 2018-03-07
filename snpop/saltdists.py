"""
Module related to details of SALT distributions
"""
from __future__ import absolute_import, print_function
__all__ = ['SALT2_MMDist']
import numpy as np

def double_gauss(mu, sigp, sigm, size):
    """Double Gaussian distribution. Note: mu is the mode and not the mean."""
    
    sigm = abs(sigm) # Just in case

    p = np.array([sigp, sigm], dtype=np.float64) # probability of one side is proportional to sigma on that side
    p /= sum(p)


    sig = np.random.choice([sigp, -sigm], size = size, replace = True, p = p)
    
    return abs(np.random.normal(size = size))*sig + mu

def SALT2_MMDist(numSN,
                 cm=-0.0474801042369, cs1=0.0965032273527, cs2=0.042844366359,
                 x1m=0.872727291354, x1s1=0.358731835038, x1s2=1.42806797468,
                 mm=10.701690617, ms1=0.334359086569, ms2=1.0750402101,
                 mBm=-19.0199168813, mc=-0.0838387899933, mt=10.,
                 cc=3.20907949118, cx1=-0.137042055737):
    """
    Generates "noise"-free mB, x1, c. Trained on JLA SALT2-4 SDSS z < 0.2 and
    SNLS z < 0.5. Very simple model with linear alpha/beta and same
    distribution irrspective of host-mass. mB needs h=0.7 distance modulus
    added to it.

    From D. Rubin
    """
    color = double_gauss(cm, cs1, cs2, size=numSN)
    x1 = double_gauss(x1m, x1s1, x1s2, size=numSN)
    mass = double_gauss(mm, ms1, ms2, size=numSN)

    mB = mBm + mc * (mass > 10.) + cc * color + cx1 * x1

    return mB, x1, color, mass
