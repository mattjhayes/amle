"""
The cross_validator_1 module runs an experiment multiple times
to provide cross validation result data
"""

from numpy import exp, array, random, dot

class Aggregator(object):
    """
    An aggregator module for import by AMLE
    """
    def __init__(self, logger, datasets, parameters, experiments):
        """
        Initialise the aggregator
        """
        self.logger = logger
        #*** Retrieve parameters passed to us:
        self.input_variables = parameters['experiments']

    def run(self,):
        """
        Run the experiment(s)
        """
        
        self.logger.debug("in run")

