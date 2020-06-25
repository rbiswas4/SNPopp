"""
Module related to details of SALT distributions
"""
from __future__ import absolute_import, print_function
__all__ = ['SALT2_MMDist', 'SALT2_SK16', 'sample_SALT2_SK16_hybrid']
import numpy as np
from tdaspop import double_gaussian, double_gaussian_pdf

def double_gauss(mu, sigp, sigm, size):
    """Double Gaussian distribution. Note: mu is the mode and not the mean."""
    
    sigm = abs(sigm) # Just in case

    # probability of one side is proportional to sigma on that side
    p = np.array([sigp, sigm], dtype=np.float64) # normalize
    p /= sum(p)


    sig = np.random.choice([sigp, -sigm], size = size, replace = True, p = p)
    
    return abs(np.random.normal(size = size))*sig + mu

def  sample_SALT2_SK16_hybrid(num_sn, model1, model2, probs=[0.5, 0.5],
                              rng=np.random.RandomState(1)):
    """
    Hybrid  model scheme meant to weight different modes of intrinsic dispersion

    Parameters
    ----------
    num_sn : int
        number of samples requested
    model1 : string
        must be an available model name in `SALT2_SK16`
    model2 : string
        must be an available model name in `SALT2_SK16`
    probs : sequence of size 2
        should have two floats. (prob of being model 1, and prob of being model2)
    rng : `np.random.RandomState`
        Instance of randomstate


    Returns
    -------
    samples : `np.ndarray` of shape (num_sn, 2)
    samples of x1 and c in  samples[:, 0] and samples[:, 1] respectively.
    """
    sk1 = SALT2_SK16.from_model_name(model1)
    sk2 = SALT2_SK16.from_model_name(model2)
    
    p1 = probs[0]
    num1 = np.int(np.floor(num_sn *p1))
    num2 = np.int(num_sn - num1)

    return np.vstack((sk1.sample(num1), sk2.sample(num2)))

class SALT2_SK16(object):
    """
    Implementation of distribution of x1, and c from  Scolnic and Kessler,
    ApJ, 2016.

    Parameters
    ----------
    csigmalow : float
        sigma for lower values of the double gaussian for c
    csigmahigh : float
        sigma for higher values of the double gaussian for c
    cmode : float
        mode of the c distribution
    x1sigmalow : float
        sigma for lower values of the double gaussian for x1
    x1sigmahigh : float
        sigma for higher values of the double gaussian for x1
    x1mode : float
        mode of the x1 distribution
    rng : Instance of `np.random.RandomState`
    """
    def __init__(self, csigmalow, csigmahigh, cmode, x1sigmalow, x1sigmahigh,
                 x1mode, rng=np.random.RandomState()):
        self.csigmalow = csigmalow
        self.csigmahigh = csigmahigh
        self.cmode = cmode
        self.x1sigmalow = x1sigmalow
        self.x1sigmahigh = x1sigmahigh
        self.x1mode = x1mode
        self.rng = rng


    @classmethod
    def from_model_name(cls, model_name, rng=np.random.RandomState(1)):
        """ Implementations for models calculated from all surveys and G10,
        C11 scatter, and lowz surveys with G10 and C11 scatter. Set these
        models by name by using:

        All g10 : 'sk16_allz_g10'
        All c11 : 'sk16_allz_c11'
        Lowz g10 : 'sk16_lowz_g10'
        Lowz c11 : 'sk16_lowz_c11'
        """
        if model_name.lower() == "sk16_allz_g10":
            cm  = -0.043
            csl = 0.052
            csh = 0.107
            xm  = 0.945
            xsl = 1.553
            xsh = 0.257

        elif model_name.lower() == "sk16_allz_c11":
            cm  = -0.062
            csl = 0.032
            csh = 0.113
            xm  = 0.938
            xsl = 1.551
            xsh = 0.26

        elif model_name.lower() == "sk16_lowz_g10":
            cm  = -0.055
            csl = 0.023
            csh = 0.150
            xm  = 0.436
            xsl = 3.118
            xsh = 0.724

        elif model_name.lower() == "sk16_lowz_c11":
            cm  = -0.069
            csl = 0.023
            csh = 0.083
            xm  = 0.436
            xsl = 3.118
            xsh = 0.724
        else:
            raise NotImplementedError(f'class method for ks16 for Model name {model_name} not implemented yet')
        return cls(csl, csh, cm, xsl, xsh, xm, rng)

    def pdf(self, x):
        """
        Evaluate the pdf at x:

        Parameters
        ----------
        x : `np.ndarray`, shape (N, 2)
            parmeters describing the population, with
            x[:, 0] = c, x[:, 1] = x1

        Returns
        -------
        pdf : `np.ndarray` of shape (N, 1)
        """
        c = x[:, 0]
        x1 = x[:, 1]
        pc = double_gaussian_pdf(c, self.cmode,
                                 self.csigmalow,
                                 self.csigmahigh)
        px1 = double_gaussian_pdf(x1, self.x1mode,
                                 self.x1sigmalow,
                                 self.x1sigmahigh)

        pdf = pc * px1

        return pdf

    def sample(self, num_sn):
        """
        Draw samples from the distribution specified by the class

        Parameters
        ----------
        num_sn : int
            number of SN

        Returns
        -------
        samples : `np.ndarray` with shape = (num_sn, 2)
        Contains the x1 samples in samples[:, 0] and c samples in samples[:, 1]
        """
        c = double_gaussian(self.cmode, self.csigmalow, self.csigmahigh, size=num_sn, rng=self.rng)
        x1 = double_gaussian(self.x1mode, self.x1sigmalow, self.x1sigmahigh, size=num_sn, rng=self.rng)
        X = np.zeros(shape=(num_sn, 2))
        X[:, 0] = x1
        X[:, 1] = c
        return X  




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
