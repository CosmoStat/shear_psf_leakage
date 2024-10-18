"""UNIT TESTS FOR RUN_OBJECT SUBPACKAGE.                                           
                                                                                
This module contains unit tests for the run_object subpackage.                     
                                                                                
"""                                                                             
                                                                                
from unittest import TestCase                                                   
                                                                                
from numpy import testing as npt                                                
import os                                                                       
import sys

from shear_psf_leakage import run_object as run

class RunObjectTestCases(TestCase):
    """Test case for the ```run_object``` module."""

    def setup(self):
        """Set test parameter values."""

        pass

    def tearDow(self):
        """Unset test parameter values."""

        pass

    def test_test(self):
        """Test of the ```test``` 2D quadratic fit function."""

        obj = run.LeakageObject()
        obj._params['test'] = True
        obj._params['verbose'] = False
        obj.run()
