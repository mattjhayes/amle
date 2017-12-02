"""
AMLE evaluate.py Unit Tests
"""

#*** Handle tests being in different directory branch to app code:
import sys
sys.path.insert(0, '../amle')

#*** Testing imports:
import unittest

#*** Logging:
import logging
import coloredlogs

#*** AMLE imports:
import evaluate as evaluate_module

#*** Set up logging:
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,
                    fmt="%(asctime)s %(module)s[%(process)d] %(funcName)s " + 
                    "%(levelname)s %(message)s",
                    datefmt='%H:%M:%S')

RESULTS_1 = [{'actual': 0.0, 'computed': 0.04},
             {'actual': 0.0, 'computed': 0.14},
             {'actual': 0.0, 'computed': 0.09999},
             {'actual': 0.0, 'computed': -0.04},
             {'actual': 0.0, 'computed': 0.0956},
             {'actual': 1.0, 'computed': 0.04},
             {'actual': 1.0, 'computed': 0.91},
             {'actual': 1.0, 'computed': 0.89},
             {'actual': 1.0, 'computed': 1.09999999999},
             {'actual': 1.0, 'computed': 1.1000000001}]
THRESHOLD_1 = 0.1

def test_simple_accuracy():
    """
    Test the simple_accuracy method.
    Note: numbers are floats so can't evaluate equality
    """
    evaluate = evaluate_module.Evaluate(logger)
    assert evaluate.simple_accuracy(RESULTS_1, THRESHOLD_1) == 60
