"""
AMLE mlnn5.py Unit Tests
"""

#*** Handle tests being in different directory branch to app code:
import sys
sys.path.insert(0, '../amle/algorithms/examples/mlnn')

#*** Testing imports:
import unittest

#*** Logging:
import logging
import coloredlogs

#*** Numpy for matrices:
import numpy as np

#*** AMLE imports:
import mlnn5 as mlnn5_module

#*** Set up logging:
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,
                    fmt="%(asctime)s %(module)s[%(process)d] %(funcName)s " + 
                    "%(levelname)s %(message)s",
                    datefmt='%H:%M:%S')

def test_loss_mse():
    """
    Test the loss_mse method.
    """
    target = np.array([(1,1,0,0),(1,1,0,0)])
    actual = np.array([(1,-1,0,0),(-1,1,0,0)])
    expected_result = np.array([[ 0.,  2.,  0.,  0.],[ 2.,  0.,  0.,  0.]])

    test_result = mlnn5_module.loss_mse(target, actual)
    
    assert np.array_equal(test_result, expected_result)

def test_sigmoid():
    """
    Test the sigmoid function
    """
    assert mlnn5_module.sigmoid(0) == 0.5

def test_sigmoid_derivative():
    """
    Test the sigmoid_derivative function
    """
    assert mlnn5_module.sigmoid_derivative(0) == 0.25

def test_tanh():
    """
    Test the tanh function
    """
    assert mlnn5_module.tanh(0) == 0

def test_tanh_derivative():
    """
    Test the tanh_derivative function
    """
    assert mlnn5_module.tanh_derivative(0) == 1
