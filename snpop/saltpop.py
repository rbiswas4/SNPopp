"""
Concrete implementations of `varpop.BaseSpatialPopulation` for SALT2 SN.
"""
from __future__ import absolute_import, print_function
__all__ = ['SimpleSALTPopulation', 'GMM_SALTPopulation']
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15
import sncosmo
from tdaspop import (BasePopulation,
                    PowerLawRates)
from . import SALT2_MMDist


class SimpleSALTPopulation(BasePopulation):
    """
    Concrete Implementation of `varpop.BasePopulation` for SALT parameters
    based on a normal distribution 
    """
    def __init__(self, zSamples, rng, snids=None, alphaTripp=0.11, betaTripp=3.14,
                 cSigma=0.1, x1Sigma=1.0, meanMB=-19.3, Mdisp=0.15,
                 cosmo=Planck15, mjdmin=59580., surveyDuration=10.):
        """
        Parameters
        ----------
        zSamples : sequence
            z values list or one-d array
        snids : sequence of integers or strings
            sequence of ids
        alphaTripp : float, defaults to 0.11
            `alpha` in the Tripp relation
        betaTripp : float, defaults to 3.14
            `beta` in the Tripp relation
        cSigma : float, defaults to 0.1
            standard deviation in `c`
        x1Sigma : float, defaults to 1.0
            standard deviation in `x1`
        meanMB : float, defaults to -19.3
            absolute BessellB magnitude of the SN at peak.
        Mdisp : float, defaults to 0.15
            intrinsic dispersion of the SN assumed to be in brightness only
        rng : instance of `np.random.RandomState`
            random state used for `model_rng`
        cosmo : instance of `astropy.cosmology`, defaults to Planck15 values
            Cosmology specified as in astropy
        mjdmin : float, units of days, defaults to 59580.0
            mjd at the start of the survey
        surveyDuration: float, units of years
            duration of the survey in years
        """
        self.zSamples = zSamples
        self._snids = snids
        self.alpha = alphaTripp
        self.beta = betaTripp
        self.cSigma = cSigma
        self.x1Sigma = x1Sigma
        self.centralMabs = meanMB
        self.Mdisp = Mdisp
        self._rng = rng
        self.cosmo = cosmo
        self._mjdmin = mjdmin
        self.surveyDuration = surveyDuration
        self._paramsTable = None


    @classmethod
    def fromSkyArea(cls, rng, snids=None, alpha=2.6e-5, beta=1.5, 
                    alphaTripp=0.11, betaTripp=3.14,
                    cSigma=0.1, x1Sigma=1.0, meanMB=-19.3, Mdisp=0.15,
                    cosmo=Planck15, mjdmin=59580., surveyDuration=10.,
                    fieldArea=10., skyFraction=None, zmax=1.4, zmin=1.0e-8,
                    numzBins=20):
        """
        Class method to use either FieldArea or skyFraction and zbins 
        (zmin, zmax, numzins) to obtain the correct number of zSamples.
        """
        pl = PowerLawRates(rng=rng, cosmo=cosmo,
                           alpha_rate=alpha, beta_rate=beta,
                           zlower=zmin, zhigher=zmax, num_bins=numzBins, zbin_edges=None,
                           survey_duration=surveyDuration, sky_area=fieldArea,
                           sky_fraction=skyFraction)

        cl = cls(pl.z_samples, rng=rng, snids=snids, alphaTripp=alphaTripp,
                 betaTripp=betaTripp, cSigma=cSigma, x1Sigma=x1Sigma,
                 meanMB=meanMB, Mdisp=Mdisp, cosmo=cosmo, mjdmin=mjdmin,
                 surveyDuration=surveyDuration)
        return cl

    @property
    def mjdmin(self):
        return self._mjdmin

    @property
    def mjdmax(self):
        return self.mjdmin + self.surveyDuration * 365.0

    @property
    def paramsTable(self):
        if self._paramsTable is None:
            timescale = self.mjdmax - self.mjdmin
            T0Vals = self.rng_model.uniform(size=self.numSources) * timescale \
                    + self.mjdmin
            cvals = self.rng_model.normal(loc=0., scale=self.cSigma,
                                          size=self.numSources)
            x1vals = self.rng_model.normal(loc=0., scale=self.x1Sigma,
                                           size=self.numSources)
            M = - self.alpha * x1vals - self.beta * cvals
            MnoDisp = self.centralMabs + M
            Mabs = MnoDisp + self.rng_model.normal(loc=0., scale=self.Mdisp,
                                               size=self.numSources)
            x0 = np.zeros(self.numSources)
            mB = np.zeros(self.numSources)
            model = sncosmo.Model(source='SALT2')
            for i, z in enumerate(self.zSamples):
                model.set(z=z, x1=x1vals[i], c=cvals[i])
                model.set_source_peakabsmag(Mabs[i], 'bessellB', 'ab',
                                            cosmo=self.cosmo)
                x0[i] = model.get('x0')
                mB[i] = model.source.peakmag('bessellB', 'ab')

            df = pd.DataFrame(dict(x0=x0, mB=mB, x1=x1vals, c=cvals,
                                   MnoDisp=MnoDisp, Mabs=Mabs, t0=T0Vals,
                                   z=self.zSamples, idx=self.idxvalues))
            df['model'] = 'SALT2'
            self._paramsTable = df.set_index('idx')
        return self._paramsTable

    @property
    def idxvalues(self):
        if self._snids is None:
            self._snids = np.arange(self.numSources)
        return self._snids

    @property
    def numSources(self):
        return len(self.zSamples)

    @property
    def rng_model(self):
        if self._rng is None:
            raise ValueError('rng must be provided')
        return self._rng

class GMM_SALTPopulation(SimpleSALTPopulation):
    
    def __init__(self, zSamples, rng, snids=None, alphaTripp=0.11, betaTripp=3.14,
                 cSigma=0.1, x1Sigma=1.0, meanMB=-19.3, Mdisp=0.15,
                 cosmo=Planck15, mjdmin=59580., surveyDuration=10.):
        super().__init__(zSamples, rng, snids=None, alphaTripp=0.11, betaTripp=3.14,
                         cSigma=0.1, x1Sigma=1.0, meanMB=-19.3, Mdisp=0.15,
                         cosmo=Planck15, mjdmin=59580., surveyDuration=10.)


    @property
    def h70cosmo(self):
        H70cosmo = self.cosmo.clone(name='H70cosmo',
                                    H0=self.cosmo.H0 * (70/self.cosmo.H0.value))
        return H70cosmo

    @property
    def paramsTable(self):
        if self._paramsTable is None:
            timescale = self.mjdmax - self.mjdmin
            T0Vals = self.rng_model.uniform(size=self.numSources) * timescale \
                    + self.mjdmin
            mBB, x1, c, m = SALT2_MMDist(self.numSources)
            mBB += self.h70cosmo.distmod(self.zSamples).value
            # cvals = self.rng_model.normal(loc=0., scale=self.cSigma,
            #                              size=self.numSources)
            # x1vals = self.rng_model.normal(loc=0., scale=self.x1Sigma,
            #                              size=self.numSources)
            # M = - self.alpha * x1vals - self.beta * cvals
            # MnoDisp = self.centralMabs + M
            MnoDisp = mBB - self.cosmo.distmod(self.zSamples).value
            Mabs = MnoDisp + self.rng_model.normal(loc=0., scale=self.Mdisp,
                                                   size=self.numSources)
            x0 = np.zeros(self.numSources)
            mB = np.zeros(self.numSources)
            model = sncosmo.Model(source='SALT2')
            for i, z in enumerate(self.zSamples):
                model.set(z=self.zSamples[i], x1=x1[i], c=c[i])
                model.set_source_peakabsmag(Mabs[i], 'bessellB', 'ab',
                                            cosmo=self.cosmo)
                x0[i] = model.get('x0')
                mB[i] = model.source.peakmag('bessellB', 'ab')

            df = pd.DataFrame(dict(x0=x0, mB=mB, x1=x1, c=c, mBB=mBB,
                                   MnoDisp=MnoDisp, Mabs=Mabs, t0=T0Vals,
                                   z=self.zSamples, idx=self.idxvalues))
            df['model'] = 'SALT2'
            self._paramsTable = df.set_index('idx')
        return self._paramsTable


###class CoordSamples(PositionSamples, HealpixTiles):
###    def __init__(self, nside, hpOpSim, rng):
###        self.nside = nside
###        self.nside = nside
###        super(self.__class__, self).__init__(nside=nside, healpixelizedOpSim=hpOpSim)
###        self._rng = rng
###    @property
###    def randomState(self):
###        if self._rng is None:
###            raise ValueError('self._rng should not be None')
###        return self._rng
###    def _angularSamples(self, phi_c, theta_c, radius, numSamples, tileID):
###        phi, theta = super(self.__class__, self).samplePatchOnSphere(phi=phi_c,
###								     theta=theta_c, delta=radius, 
###                                                                     size=numSamples, rng=self.randomState)
###        tileIds = hp.ang2pix(nside=self.nside, theta=np.radians(theta),
###			     phi=np.radians(phi), nest=True)
###        inTile = tileIds == tileID
###        return phi[inTile], theta[inTile]
###        
###    def positions(self, tileID, numSamples):
###        res_phi = np.zeros(numSamples)
###        res_theta = np.zeros(numSamples)
###        ang = hp.pix2ang(nside=self.nside, ipix=tileID, nest=True)
###        radius = np.degrees(np.sqrt(hp.nside2pixarea(self.nside) / np.pi))
###        phi_c, theta_c = np.degrees(ang[::-1])
###        num_already = 0
###        while numSamples > 0:
###            phi, theta = self._angularSamples(phi_c, theta_c, radius=2*radius,
###					      numSamples=numSamples,
###					      tileID=tileID)
###            num_obtained = len(phi)
###            res_phi[num_already:num_obtained + num_already] = phi
###            res_theta[num_already:num_obtained + num_already] = theta
###            num_already += num_obtained
###            numSamples -= num_obtained
###        return res_phi, res_theta
###
###
###
###class TwinklesRates(PowerLawRates):
###    def __init__(self, galsdf, rng, alpha=2.6e-3, beta=1.5, zbinEdges=None,
###                 zlower=0.0000001, zhigher=1.2, numBins=24, agnGalids=None,
###                 surveyDuration=10., fieldArea=None, skyFraction=None,
###                 cosmo=Planck15):
###        PowerLawRates.__init__(self, rng=rng, alpha=alpha, beta=beta,
###                               zbinEdges=zbinEdges, zlower=zlower,
###                               zhigher=zhigher, numBins=numBins,
###                               fieldArea=fieldArea, cosmo=cosmo)
###        self._galsdf = galsdf
###        if agnGalids is None:
###            agnGalids = []
###        self.agnGalTileIds = tuple(agnGalids)
###        self.binWidth = np.diff(self.zbinEdges)[0]
###        #self.galsdf =None
###        self.binnedGals = None
###        self.numGals = None
###        self.gdf = None
###        self.rng = rng
###        self._selectedGals = None
###        
###    
###    @property
###    def galsdf(self):
###        if self.gdf is not None:
###            return self.gdf
###        zhigher = self.zhigher
###        self.addRedshiftBins()
###        
###        vetoedGaltileIds = tuple(self.agnGalTileIds)
###        sql_query = 'redshift <= @zhigher and galtileid not in @vetoedGaltileIds'
###        galsdf = self._galsdf.query(sql_query)
###        self.binnedGals = galsdf.groupby('redshiftBin')
###        self.numGals = self.binnedGals.redshift.count()
###        galsdf['probHist'] = galsdf.redshiftBin.apply(self.probHost)
###        galsdf['hostAssignmentRandom'] = self.rng.uniform(size=len(galsdf))
###        self.gdf = galsdf
###        return galsdf
###    
###    def addRedshiftBins(self):
###        self._galsdf['redshiftBin'] = (self._galsdf.redshift - self.zlower) // self.binWidth
###        self._galsdf.redshiftBin = self._galsdf.redshiftBin.astype(np.int)
###        
###    @property    
###    def selectedGals(self):
###        if self._selectedGals is None:
###            df = self.galsdf.query('hostAssignmentRandom < probHist')
###            df.galtileid = df.galtileid.astype(int)
###            df['snid'] = df.id
###        else:
###            df = self._selectedGals
###        return df
###    def probHost(self, binind):
###        return np.float(self.numSN()[binind]) / self.numGals[binind]
###    
###    @property
###    def zSamples(self):
###        return self.selectedGals.redshift.values
###
###class CatSimPositionSampling(object):
###    def __init__(self, rng, galdf, snAngularUnits='degrees'):
###        self.galdf = galdf.copy()
###        self.rng = rng
###        self.ss = SersicSamples(rng=self.rng)
###        self.radianOverArcSec = np.pi / 180.0 / 3600.
###        self.snAngularUnits=snAngularUnits
###
###    
###    def f1(self, x):
###        return self.ss.sampleRadius(x)[0]
###    def f4(self, x):
###        return self.ss.sampleRadius(x, sersicIndex=4)[0]
###    
###    def SampleDiskAngles(self, x):
###        return self.ss.sampleAngles(x.a_d, x.b_d)[0]
###    def SampleBulgeAngles(self, x):
###        return self.ss.sampleAngles(x.a_b, x.b_b)[0]
###    @staticmethod
###    def theta(df, angle='diskAngle', PositionAngle='pa_disk'):
###        return np.radians(df[angle] - df[PositionAngle] + 90.)
###
###    @staticmethod
###    def snInDisk(x, value=" None"):
###        if x.sedFilenameDisk == value:
###            return 0
###        elif x.sedFilenameBulge == value:
###            return 1
###        else:
###            return np.random.choice([0, 1], p=[0.5, 0.5])
###    def addPostions(self):
###        self.galdf['isinDisk'] = self.galdf.apply(self.snInDisk, axis=1)
###        self.galdf['bulgeRadialPos'] = self.galdf.BulgeHalfLightRadius.apply(self.f4)
###        self.galdf['diskRadialPos'] = self.galdf.DiskHalfLightRadius.apply(self.f1)
###        self.galdf['bulgeAngle'] = self.galdf.apply(self.SampleBulgeAngles, axis=1)
###        self.galdf['diskAngle'] = self.galdf.apply(self.SampleDiskAngles, axis=1)
###        self.galdf['DeltaRaDisk'] = self.galdf.diskRadialPos * np.cos(self.theta(self.galdf)) * self.galdf.isinDisk
###        self.galdf['DeltaRaBulge'] = self.galdf.bulgeRadialPos * self.theta(self.galdf, angle='bulgeAngle', 
###                                                                            PositionAngle='pa_bulge')\
###                                     * (1 - self.galdf.isinDisk)
###        self.galdf['DeltaDecDisk'] = self.galdf.diskRadialPos * np.sin(self.theta(self.galdf)) * self.galdf.isinDisk
###        self.galdf['DeltaDecBulge'] = self.galdf.bulgeRadialPos * np.sin(self.theta(self.galdf, angle='bulgeAngle',
###                                                                                 PositionAngle='pa_bulge')) \
###                                                                                 * (1 - self.galdf.isinDisk)
###        self.galdf['snra'] = self.radianOverArcSec *\
###            self.galdf[['DeltaRaDisk', 'DeltaRaBulge']].apply(np.nansum, axis=1)\
###            + self.galdf.raJ2000
###
###        self.galdf['sndec'] = self.radianOverArcSec *\
###            self.galdf[['DeltaDecDisk', 'DeltaDecBulge']].apply(np.nansum, axis=1)\
###            + self.galdf.decJ2000
###
###        if self.snAngularUnits == 'degrees':
###            self.galdf[['snra', 'sndec']] = \
###                    self.galdf[['snra', 'sndec']].apply(np.degrees)
###        elif self.snAngularUnits =='radians':
###            pass
###        else :
###            raise NotImplementedError('conversion to snAngularUnits {} not implemented', self.snAngularUnits)
###
###class TwinklesSim(TwinklesRates):
###
###    def __init__(self, catsimgaldf, rng, fieldArea, cosmo, agnGalids=None, numBins=24,
###                 rate_alpha=0.0026, rate_beta=1.5, zlower=1.0e-7, zhigher=1.2,
###                 zbinEdges=None, tripp_alpha=0.11, tripp_beta=3.14, mjdmin=0.):
###        super(TwinklesSim, self).__init__(catsimgaldf, rng=rng,
###                                          cosmo=cosmo, fieldArea=fieldArea,
###                                          agnGalids=agnGalids,
###                                          alpha=rate_alpha, beta=rate_beta,
###                                          zlower=zlower, zhigher=1.2,
###                                          numBins=numBins, zbinEdges=None,
###                                          skyFraction=None)
###        self.cosmo = cosmo
###        self.beta_rate = deepcopy(self.beta)
###        self.alpha_rate = deepcopy(self.alpha)
###        self.numSN = len(self.zSamples)
###        self.tripp_alpha = tripp_alpha
###        self.tripp_beta = tripp_beta
###        self.catsimpos = CatSimPositionSampling(rng=self.rng, 
###                                                galdf=self.selectedGals)
###        self.catsimpos.addPostions()
###        self.mjdmin = mjdmin
###        self.salt2params = SimpleSALTDist(numSN=self.numSN, 
###                                          zSamples=self.zSamples, 
###                                          rng=self.rng,
###                                          cosmo=self.cosmo,
###                                          snids=self.catsimpos.galdf.snid,
###                                          alpha=self.tripp_alpha,
###                                          beta=self.tripp_beta,
###                                          surveyDuration=10.,
###                                          mjdmin=self.mjdmin)
###        self._snparamdf = None
###
###    @property
###    def snparamdf(self):
###        """
###        dataFrame with information about the object
###        """
###        if self._snparamdf is None:
###            self.salt2params.paramSamples.set_index('snid', inplace=True)
###            self.catsimpos.galdf.set_index('snid', inplace=True)
###            self._snparamdf = self.salt2params.paramSamples.join(self.catsimpos.galdf)
###        return self._snparamdf
