"""
This module allows to compute a theoretical estimate of the covariance matrix of the tau statistics
to estimate systematic error from the PSF using rho and tau statistics.
Author: Sacha Guerrini
"""

import numpy as np
from astropy.io import fits

import scipy.integrate as integrate
import scipy.interpolate as interpolate

import treecorr

class CovTauTh:
    """
    CovTauTh

    Class to build and compute the covariance matrix of the tau statistics using a theoretical estimate
    """

    def __init__(
        self,
        path_gal,
        path_psf,
        hdu_psf,
        treecorr_config,
        A,
        n_e,
        n_psf,
        params=None,
        **kwargs):
        """
        Initalize the CovTauTh class.
        Galaxy and PSF star catalogs are used to initialize the value of the different shape noise.

        Parameters
        ----------
        path_gal: str
            Path to the galaxy catalog
        path_psf: str
            Path to the PSF catalog
        hdu_psf: int
            HDU of the PSF catalog
        treecorr_config: dict
            Configuration of the treecorr catalogs
        A: float
            Area of the survey
        n_e: float
            Number density of galaxies
        n_psf: float
            Number density of PSF stars
        params: dict
            Parameters of the class
        """
        self.treecorr_config = treecorr_config
        self.A = A
        self.n_e = n_e
        self.n_psf = n_psf

        if params is None:
            self.params_default()
        else:
            self.set_params(params)

        #Load the catalogs
        cat_gal, cat_psf = fits.getdata(path_gal), fits.open(path_psf)[hdu_psf].data

        #Create treecorr catalogs
        self.gal, self.psf, self.psf_error, self.size_error = self.create_treecorr_catalog(cat_gal, cat_psf)

        #Compute the shape noise
        self.sigma_e = self.compute_shape_noise(self.gal, self.gal)
        self.sigma_psf = self.compute_shape_noise(self.psf, self.psf)
        self.sigma_psf_error = self.compute_shape_noise(self.psf_error, self.psf_error)
        self.sigma_psf_size_error = self.compute_shape_noise(self.size_error, self.size_error)


        #interpolate rho stats
        self.rho_0_p_itp = self.build_interpolator(self.psf, self.psf, type='plus')
        self.rho_0_m_itp = self.build_interpolator(self.psf, self.psf, type='minus')
        self.rho_1_p_itp = self.build_interpolator(self.psf_error, self.psf_error, type='plus')
        self.rho_1_m_itp = self.build_interpolator(self.psf_error, self.psf_error, type='minus')
        self.rho_2_p_itp = self.build_interpolator(self.psf, self.psf_error, type='plus')
        self.rho_2_m_itp = self.build_interpolator(self.psf, self.psf_error, type='minus')
        self.rho_3_p_itp = self.build_interpolator(self.size_error, self.size_error, type='plus')
        self.rho_3_m_itp = self.build_interpolator(self.size_error, self.size_error, type='minus')
        self.rho_4_p_itp = self.build_interpolator(self.psf_error, self.size_error, type='plus')
        self.rho_4_m_itp = self.build_interpolator(self.psf_error, self.size_error, type='minus')
        self.rho_5_p_itp = self.build_interpolator(self.psf, self.size_error, type='plus')
        self.rho_5_m_itp = self.build_interpolator(self.psf, self.size_error, type='minus')


        #interpolate tau stats
        self.tau_0_p_itp = self.build_interpolator(self.gal, self.psf, type='plus')
        self.tau_0_m_itp = self.build_interpolator(self.gal, self.psf, type='minus')
        self.tau_2_p_itp = self.build_interpolator(self.gal, self.psf_error, type='plus')
        self.tau_2_m_itp = self.build_interpolator(self.gal, self.psf_error, type='minus')
        self.tau_5_p_itp = self.build_interpolator(self.gal, self.size_error, type='plus')
        self.tau_5_m_itp = self.build_interpolator(self.gal, self.size_error, type='minus')

        #interpolate xi plus and minus
        self.xi_plus_itp = self.build_interpolator(self.gal, self.gal, type='plus')
        self.xi_minus_itp = self.build_interpolator(self.gal, self.gal, type='minus')

        self.rho_stats = kwargs.get("rho_stats", None)
        self.tau_stats = kwargs.get("tau_stats", None)
        self.xi_plus = kwargs.get("xi_plus", None)
        self.xi_minus = kwargs.get("xi_minus", None)

        dummy_cat = treecorr.Catalog(ra=[0], dec=[0], g1=[0], g2=[0], w=[1], ra_units='deg', dec_units='deg')
        gg = treecorr.GGCorrelation(self.treecorr_config)
        gg.process(dummy_cat, dummy_cat)
        self.bins = gg.meanr

        self.component_dict = {
            '00': {
                'eb': {'p': self.tau_0_p_itp, 'm': self.tau_0_m_itp},
                'ec': {'p': self.tau_0_p_itp, 'm': self.tau_0_m_itp},
                'bc': {'p': self.rho_0_p_itp, 'm': self.rho_0_m_itp},
                'sigma_bc': None,
            },
            '22': {
                'eb': {'p': self.tau_2_p_itp, 'm': self.tau_2_m_itp},
                'ec': {'p': self.tau_2_p_itp, 'm': self.tau_2_m_itp},
                'bc': {'p': self.rho_1_p_itp, 'm': self.rho_1_m_itp},
                'sigma_bc': None,
            },
            '55': {
                'eb': {'p': self.tau_5_p_itp, 'm': self.tau_5_m_itp},
                'ec': {'p': self.tau_5_p_itp, 'm': self.tau_5_m_itp},
                'bc': {'p': self.rho_3_p_itp, 'm': self.rho_3_m_itp},
                'sigma_bc': None,
            },
            '02': {
                'eb': {'p': self.tau_0_p_itp, 'm': self.tau_0_m_itp},
                'ec': {'p': self.tau_2_p_itp, 'm': self.tau_2_m_itp},
                'bc': {'p': self.rho_2_p_itp, 'm': self.rho_2_m_itp},
            },
            '05': {
                'eb': {'p': self.tau_0_p_itp, 'm': self.tau_0_m_itp},
                'ec': {'p': self.tau_5_p_itp, 'm': self.tau_5_m_itp},
                'bc': {'p': self.rho_5_p_itp, 'm': self.rho_5_m_itp},
            },
            '25': {
                'eb': {'p': self.tau_2_p_itp, 'm': self.tau_2_m_itp},
                'ec': {'p': self.tau_5_p_itp, 'm': self.tau_5_m_itp},
                'bc': {'p': self.rho_4_p_itp, 'm': self.rho_4_m_itp},
            }
        }

    def params_default(self):
        """
        Params Default.

        Initialize the parameters of the class with columns name from SPV1.
        For the treecorr configuration, default parameters are:
        -coord_units: degree
        -sep_units: arcmin
        -theta_min: 0.1
        -theta_max: 250
        -n_theta: 20
        """

        self._params = {
            "e1_col": "e1",
            "e2_col": "e2",
            "w_col": "w",
            "ra_col": "RA",
            "dec_col": "Dec",
            "e1_PSF_col": "E1_PSF_HSM",
            "e2_PSF_col": "E2_PSF_HSM",
            "e1_star_col": "E1_STAR_HSM",
            "e2_star_col": "E2_STAR_HSM",
            "PSF_size": "SIGMA_PSF_HSM",
            "star_size": "SIGMA_STAR_HSM",
            "PSF_flag": "FLAG_PSF_HSM",
            "star_flag": "FLAG_STAR_HSM",
            "R11": np.array([1]),
            "R22": np.array([1]),
            "square_size": True,
            "ra_units": "deg",
            "dec_units": "deg"
        }
    
    def set_params(self, params):
        """
        set_params:

        Initialize the parameters to the given configuration. Can also update the current params.
        """
        self._params = params

    def create_treecorr_catalog(self, cat_gal, cat_psf):
        """
        Create treecorr catalogs from the galaxy and PSF catalogs to compute estimate shape noise in covariance terms.
        Works only for shapepipe catalogs right now. A config file will be added later
        
        Parameters
        ----------
        cat_gal: numpy.ndarray
            Galaxy catalog
        cat_psf: numpy.ndarray
            PSF catalog
        
        Returns
        -------
        treecorr.Catalog, treecorr.Catalog, treecorr.Catalog, treecorr.Catalog
            Galaxy catalog, PSF catalog, PSF error catalog, PSF size error catalog
        """
        ra_units = self.treecorr_config.get('ra_units', 'deg')
        dec_units = self.treecorr_config.get('dec_units', 'deg')
        if isinstance(self._params['R11'], str):
            e1_gal = (cat_gal[self._params['e1_col']]-np.average(cat_gal[self._params['e1_col']], weights=cat_gal[self._params['w_col']]))/np.average(cat_gal[self._params['R11']])
            e2_gal = (cat_gal[self._params['e2_col']]-np.average(cat_gal[self._params['e2_col']], weights=cat_gal[self._params['w_col']]))/np.average(cat_gal[self._params['R22']])
        else:
            e1_gal = (cat_gal[self._params['e1_col']]-np.average(cat_gal[self._params['e1_col']], weights=cat_gal[self._params['w_col']]))
            e2_gal = (cat_gal[self._params['e2_col']]-np.average(cat_gal[self._params['e2_col']], weights=cat_gal[self._params['w_col']]))
        gal = treecorr.Catalog(
            ra=cat_gal[self._params['ra_col']], dec=cat_gal[self._params['dec_col']],
            w=cat_gal[self._params['w_col']], g1=e1_gal, g2=e2_gal,
            ra_units=ra_units, dec_units=dec_units
        )

        psf = treecorr.Catalog(
            ra=cat_psf[self._params['ra_col']], dec=cat_psf[self._params['dec_col']],
            g1=cat_psf[self._params['e1_PSF_col']], g2=cat_psf[self._params['e2_PSF_col']],
            ra_units=ra_units, dec_units=dec_units
        )

        psf_error = treecorr.Catalog(
            ra=cat_psf[self._params['ra_col']], dec=cat_psf[self._params['dec_col']],
            g1=cat_psf[self._params['e1_star_col']]-cat_psf[self._params['e1_PSF_col']],
            g2=cat_psf[self._params['e2_star_col']]-cat_psf[self._params['e2_PSF_col']],
            ra_units=ra_units, dec_units=dec_units
        )

        size_resid = (cat_psf[self._params['star_size']]**2-cat_psf[self._params['PSF_size']]**2)/cat_psf[self._params['star_size']]**2\
        if self._params['square_size'] else\
        (cat_psf[self._params['star_size']]-cat_psf[self._params['PSF_size']])/cat_psf[self._params['star_size']]

        size_error = treecorr.Catalog(
            ra=cat_psf[self._params['ra_col']], dec=cat_psf[self._params['dec_col']],
            g1=cat_psf[self._params['e1_star_col']]*size_resid,
            g2=cat_psf[self._params['e2_star_col']]*size_resid,
            ra_units=ra_units, dec_units=dec_units
        )
        return gal, psf, psf_error, size_error
    
    def compute_shape_noise(self, cat1, cat2):
        """
        Compute the shape noise between two catalogs

        Parameters
        ----------
        cat1: treecorr.Catalog
            First catalog
        cat2: treecorr.Catalog
            Second catalog
        
        Returns
        -------
        float
            Shape noise between the two catalogs
        """
        return 0.5*np.average((cat1.g1*cat2.g1)+(cat1.g2*cat2.g2), weights=cat1.w*cat2.w)
    
    def build_interpolator(self, cat_1, cat_2, type):
        """
        Build an interpolator for the given catalogs on a sufficient range.
        Scipy interpolator will be used between the bins. The function will then return 0 outside
        this range.

        Parameters
        ----------
        cat_1: treecorr.Catalog
            First catalog to be cross-correlated
        cat_2: treecorr.Catalog
            Second catalog to be cross-correlated
        type: str
            Type of the function to interpolate ('plus' or 'minus')

        Returns
        -------
        callable:
            Interpolator function
        """
        assert (type in ['plus', 'minus']), ("The type must be either 'plus' or 'minus'")

        gg = treecorr.GGCorrelation(nbins=30, min_sep=1e-3, max_sep=1e3, sep_units='arcmin')
        gg.process(cat_1, cat_2)
        bins = gg.meanr
        if type=='plus':
            values = gg.xip
        elif type=='minus':
            values = gg.xim
        interpolator = interpolate.make_interp_spline(bins, values, k=1)
        func = lambda x: np.where((x >= bins[0]) & (x <= bins[-1]),
                                  interpolator(x),
                                np.where(x<bins[0], np.zeros_like(x), #np.where(x < bins[0], np.ones_like(x)*values[0],
                                np.zeros_like(x))
        )
        return func

    def build_cov(self, **kwargs):
        """
        Computes the covariance matrix of the tau statistics using a theoretical estimate

        Returns
        -------
        numpy.ndarray
            Covariance matrix of the tau statistics
        """
        cov_00 = self.compute_00_comp(**kwargs)
        cov_02 = self.compute_02_comp(**kwargs)
        cov_05 = self.compute_05_comp(**kwargs)
        cov_25 = self.compute_25_comp(**kwargs)
        cov_22 = self.compute_22_comp(**kwargs)
        cov_55 = self.compute_55_comp(**kwargs)

        cov = np.block([
            [cov_00, cov_02, cov_05],
            [cov_02.T, cov_22, cov_25],
            [cov_05.T, cov_25.T, cov_55]
        ])
        return cov

    def compute_00_comp(self, **kwargs):
        """
        Computes the auto-correlation tau_0/tau_0.

        Returns
        -------
        numpy.ndarray
            Auto-correlation tau_0/tau_0
        """
        cov_00 = self.compute_sn('00', **kwargs) + self.compute_mt('00', **kwargs) + self.compute_cv_rho_plus('00', **kwargs) +\
            self.compute_cv_tau_plus('00', **kwargs) + self.compute_cv_rho_minus('00', **kwargs) +\
            self.compute_cv_tau_minus('00', **kwargs)
        return cov_00

    def compute_02_comp(self, **kwargs):
        """
        Computes the cross-correlation tau_0/tau_2.

        Returns
        -------
        numpy.ndarray
            Cross-correlation tau_0/tau_2
        """
        cov_02 = self.compute_sn('02', **kwargs) + self.compute_mt('02', **kwargs) + self.compute_cv_rho_plus('02', **kwargs) +\
            self.compute_cv_tau_plus('02', **kwargs) + self.compute_cv_rho_minus('02', **kwargs) +\
            self.compute_cv_tau_minus('02', **kwargs)
        return cov_02

    def compute_05_comp(self, **kwargs):
        """
        Computes the cross-correlation tau_0/tau_5.

        Returns
        -------
        numpy.ndarray
            Cross-correlation tau_0/tau_5
        """
        cov_05 = self.compute_sn('05', **kwargs) + self.compute_mt('05', **kwargs) + self.compute_cv_rho_plus('05', **kwargs) +\
            self.compute_cv_tau_plus('05', **kwargs) + self.compute_cv_rho_minus('05', **kwargs) +\
            self.compute_cv_tau_minus('05', **kwargs)
        return cov_05

    def compute_25_comp(self, **kwargs):
        """
        Computes the cross-correlation tau_2/tau_5.

        Returns
        -------
        numpy.ndarray
            Cross-correlation tau_2/tau_5
        """
        cov_25 = self.compute_sn('25', **kwargs) + self.compute_mt('25', **kwargs) + self.compute_cv_rho_plus('25', **kwargs) +\
            self.compute_cv_tau_plus('25', **kwargs) + self.compute_cv_rho_minus('25', **kwargs) +\
            self.compute_cv_tau_minus('25', **kwargs)
        return cov_25

    def compute_22_comp(self, **kwargs):
        """
        Computes the auto-correlation tau_2/tau_2.

        Returns
        -------
        numpy.ndarray
            Auto-correlation tau_2/tau_2
        """
        cov_22 = self.compute_sn('22', **kwargs) + self.compute_mt('22', **kwargs) +self.compute_cv_rho_plus('22', **kwargs) +\
            self.compute_cv_tau_plus('22', **kwargs) + self.compute_cv_rho_minus('22', **kwargs) +\
            self.compute_cv_tau_minus('22', **kwargs)
        return cov_22

    def compute_55_comp(self, **kwargs):
        """
        Computes the auto-correlation tau_5/tau_5.

        Returns
        -------
        numpy.ndarray
            Auto-correlation tau_5/tau_5
        """
        cov_55 = self.compute_sn('55', **kwargs) + self.compute_mt('55', **kwargs) + self.compute_cv_rho_plus('55', **kwargs) +\
            self.compute_cv_tau_plus('55', **kwargs) + self.compute_cv_rho_minus('55', **kwargs) +\
            self.compute_cv_tau_minus('55', **kwargs)
        return cov_55
    
    def compute_sn(self, component, **kwargs):
        """
        Computes the shot noise component of the covariance matrix for the given component.

        Parameters
        ----------
        component: str
            Component of the covariance matrix
        
        Returns
        -------
        numpy.ndarray
            Shot noise component of the covariance matrix
        """
        assert (component in self.component_dict.keys()), ("The component must be in the following list: %s" % self.component_dict.keys())
        

        sn = np.zeros((len(self.bins), len(self.bins)))
        tau = treecorr.GGCorrelation(
                nbins=self.treecorr_config['nbins'],
                min_sep=self.treecorr_config['min_sep'],
                max_sep=self.treecorr_config['max_sep'],
                sep_units=self.treecorr_config['sep_units']
        )
        if component == '00':
            tau.process(self.gal, self.psf)
            sn = tau.cov[:len(self.bins), :len(self.bins)]
        elif component == '22':
            tau.process(self.gal, self.psf_error)
            sn = tau.cov[:len(self.bins), :len(self.bins)]
        elif component == '55':
            tau.process(self.gal, self.size_error)
            sn = tau.cov[:len(self.bins), :len(self.bins)]
        return sn

    def compute_mt(self, component, **kwargs):
        """
        Computes the mixed term component of the covariance matrix for the given component.

        Parameters
        ----------
        component: str
            Component of the covariance matrix
        
        Returns
        -------
        numpy.ndarray
            Shot noise component of the covariance matrix
        """
        assert (component in self.component_dict.keys()), ("The component must be in the following list: %s" % self.component_dict.keys())
        correlator_bc = self.component_dict[component]['bc']
        sigma_bc = self.component_dict[component].get('sigma_bc', None)
        nbin_ang = kwargs.get("nbin_ang", 20)

        mt = np.zeros((len(self.bins), len(self.bins)))
        interpolator_rho = correlator_bc['p']
        if sigma_bc is not None:
            interpolator_xi_plus = self.xi_plus_itp
        phi = np.linspace(0, np.pi, nbin_ang)
        for i in range(len(self.bins)):
            for j in range(len(self.bins)):
                y = np.sqrt(self.bins[i]**2+self.bins[j]**2-2*self.bins[i]*self.bins[j]*np.cos(phi))
                mt[i, j] = integrate.simps(interpolator_rho(y), phi)*self.sigma_e/(2*np.pi*self.A*self.n_e)
                if sigma_bc is not None:
                    mt[i, j] += integrate.simps(interpolator_xi_plus(y), phi)*sigma_bc/(2*np.pi*self.A*self.n_e)
                mt[j, i] = mt[i, j]
        return mt
    
    def compute_cv_rho_plus(self, component, **kwargs):
        """
        Computes the cosmic variance component of the covariance matrix for the given components
        based on the rho statistics + component.

        Parameters
        ----------
        components: str
            Component of the covariance matrix
        
        Returns
        -------
        numpy.ndarray
            Cosmic variance component of the covariance matrix
        """
        assert (component in self.component_dict.keys()), ("The component must be in the following list: %s" % self.component_dict.keys())
        rho = self.component_dict[component]['bc']
        nbin_ang = kwargs.get("nbin_ang", 20)
        nbin_rad = kwargs.get("nbin_rad", 20)

        cv = np.zeros((len(self.bins), len(self.bins)))

        
        interpolator_xi = self.xi_plus_itp
        interpolator_rho = rho['p']
        phi_angle = np.linspace(0, np.pi, nbin_ang)
        phi_radius = np.linspace(1e-1, 250, nbin_rad)
        for i in range(len(self.bins)):
            for j in range(len(self.bins)):
                radius_val = np.zeros(len(phi_radius))
                for k, phi_r in enumerate(phi_radius):
                    #Compute i,j term
                    y_m = np.sqrt(phi_r**2+self.bins[i]**2-2*phi_r*self.bins[i]*np.cos(phi_angle))
                    y_p = np.sqrt(phi_r**2+self.bins[j]**2+2*phi_r*self.bins[j]*np.cos(phi_angle))
                    int_xi_m = integrate.simps(interpolator_xi(y_m), phi_angle)
                    int_xi_p = integrate.simps(interpolator_xi(y_p), phi_angle)
                    int_rho_m = integrate.simps(interpolator_rho(y_m), phi_angle)
                    int_rho_p = integrate.simps(interpolator_rho(y_p), phi_angle)
                    radius_val[k] = 0.5*(int_xi_m*int_rho_p*phi_r+int_xi_p*int_rho_m*phi_r)
                cv[i, j] = integrate.simps(radius_val, phi_radius)*1/(np.pi*self.A)
        return cv

    def compute_cv_tau_plus(self, component, **kwargs):
        """
        Computes the cosmic variance component of the covariance matrix for the given components
        based on the tau statistics + component.

        Parameters
        ----------
        components: str
            Component of the covariance matrix
        
        Returns
        -------
        numpy.ndarray
            Cosmic variance component of the covariance matrix
        """
        assert (component in self.component_dict.keys()), ("The component must be in the following list: %s" % self.component_dict.keys())
        tau_b = self.component_dict[component]['eb']
        tau_c = self.component_dict[component]['ec']
        nbin_ang = kwargs.get("nbin_ang", 20)
        nbin_rad = kwargs.get("nbin_rad", 20)

        cv = np.zeros((len(self.bins), len(self.bins)))

        interpolator_tau_b = tau_b['p']
        interpolator_tau_c = tau_c['p']
        phi_angle = np.linspace(0, np.pi, nbin_ang)
        phi_radius = np.linspace(1e-1, 250, nbin_rad)
        for i in range(len(self.bins)):
            for j in range(len(self.bins)):
                radius_val = np.zeros(len(phi_radius))
                for k, phi_r in enumerate(phi_radius):
                    y_tau_b = np.sqrt(phi_r**2+self.bins[i]**2-2*phi_r*self.bins[i]*np.cos(phi_angle))
                    y_tau_c = np.sqrt(phi_r**2+self.bins[j]**2+2*phi_r*self.bins[j]*np.cos(phi_angle))
                    int_tau_b = integrate.simps(interpolator_tau_b(y_tau_b), phi_angle)
                    int_tau_c = integrate.simps(interpolator_tau_c(y_tau_c), phi_angle)
                    radius_val[k] = int_tau_b*int_tau_c*phi_r
                cv[i, j] = integrate.simps(radius_val, phi_radius)*1/(np.pi*self.A)
        return cv
    
    def compute_cv_rho_minus(self, component, **kwargs):
        """
        Computes the cosmic variance component of the covariance matrix for the given components
        based on the rho statistics - component.

        Parameters
        ----------
        components: str
            Component of the covariance matrix

        Returns
        -------
        numpy.ndarray
            Cosmic variance component of the covariance matrix
        """
        assert(component in self.component_dict.keys()), ("The component must be in the following list: %s" % self.component_dict.keys())
        rho = self.component_dict[component]['bc']
        nbin_ang = kwargs.get("nbin_ang", 20)
        nbin_rad = kwargs.get("nbin_rad", 20)

        cv = np.zeros((len(self.bins), len(self.bins)))
        #interpolator_xi = interpolate.make_interp_spline(self.bins, xi_minus, k=1)
        #interpolator_rho = interpolate.make_interp_spline(self.bins, rho, k=1)
        interpolator_xi = self.xi_minus_itp
        interpolator_rho = rho['m']
        phi_angle = np.linspace(0, np.pi, nbin_ang)
        phi_radius = np.linspace(1e-1, 250, nbin_rad)
        for i in range(len(self.bins)):
            for j in range(len(self.bins)):
                radius_val = np.zeros(len(phi_radius))
                for k, phi_r in enumerate(phi_radius):
                    psi_a = np.array([
                        phi_r-self.bins[i]*np.cos(phi_angle),
                        -self.bins[i]*np.sin(phi_angle)
                    ])
                    psi_b = np.array([
                        phi_r+self.bins[j]*np.cos(phi_angle),
                        self.bins[j]*np.sin(phi_angle)
                    ])
                    norm_a = np.linalg.norm(psi_a, axis=0)
                    polar_a = np.arctan2(psi_a[1], psi_a[0])
                    norm_b = np.linalg.norm(psi_b, axis=0)
                    polar_b = np.arctan2(psi_b[1], psi_b[0])
                    #The term coming from the sine will integrate out to 0 so we can only compute terms with cosine
                    int_xi_m = integrate.simps(interpolator_xi(norm_a)*np.cos(4*polar_a), phi_angle)
                    int_xi_p = integrate.simps(interpolator_xi(norm_b)*np.cos(4*polar_b), phi_angle)
                    int_rho_m = integrate.simps(interpolator_rho(norm_a)*np.cos(4*polar_a), phi_angle)
                    int_rho_p = integrate.simps(interpolator_rho(norm_b)*np.cos(4*polar_b), phi_angle)
                    radius_val[k] = 0.5*(int_xi_m*int_rho_p*phi_r+int_xi_p*int_rho_m*phi_r)
                cv[i, j] = integrate.simps(radius_val, phi_radius)*1/(np.pi*self.A)
        return cv

    def compute_cv_tau_minus(self, component, **kwargs):
        """
        Computes the cosmic variance component of the covariance matrix for the given components
        based on the tau statistics - component.

        Parameters
        ----------
        components: str
            Component of the covariance matrix
        
        Returns
        -------
        numpy.ndarray
            Cosmic variance component of the covariance matrix
        """
        assert (component in self.component_dict.keys()), ("The component must be in the following list: %s" % self.component_dict.keys())
        tau_b = self.component_dict[component]['eb']
        tau_c = self.component_dict[component]['ec']
        nbin_ang = kwargs.get("nbin_ang", 20)
        nbin_rad = kwargs.get("nbin_rad", 20)

        cv = np.zeros((len(self.bins), len(self.bins)))
        interpolator_tau_b = tau_b['m']
        interpolator_tau_c = tau_c['m']
        phi_angle = np.linspace(0, np.pi, nbin_ang)
        phi_radius = np.linspace(1e-1, 250, nbin_rad)
        for i in range(len(self.bins)):
            for j in range(len(self.bins)):
                radius_val = np.zeros(len(phi_radius))
                for k, phi_r in enumerate(phi_radius):
                    psi_a = np.array([
                        phi_r-self.bins[i]*np.cos(phi_angle),
                        -self.bins[i]*np.sin(phi_angle)
                    ])
                    psi_b = np.array([
                        phi_r+self.bins[j]*np.cos(phi_angle),
                        self.bins[j]*np.sin(phi_angle)
                    ])
                    norm_a = np.linalg.norm(psi_a, axis=0)
                    polar_a = np.arctan2(psi_a[1], psi_a[0])
                    norm_b = np.linalg.norm(psi_b, axis=0)
                    polar_b = np.arctan2(psi_b[1], psi_b[0])
                    #The term coming from the sine will integrate out to 0 so we can only compute terms with cosine
                    int_tau_b = integrate.simps(interpolator_tau_b(norm_a)*np.cos(4*polar_a), phi_angle)
                    int_tau_c = integrate.simps(interpolator_tau_c(norm_b)*np.cos(4*polar_b), phi_angle)
                    radius_val[k] = int_tau_b*int_tau_c*phi_r
                cv[i, j] = integrate.simps(radius_val, phi_radius)*1/(np.pi*self.A)
        return cv



                


        
