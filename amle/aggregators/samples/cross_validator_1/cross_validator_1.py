"""
The cross_validator_1 module runs an experiment multiple times
to provide cross validation result data
"""

from __future__ import division

class Aggregator(object):
    """
    An aggregator module for import by AMLE
    """
    def __init__(self, logger, datasets, parameters):
        """
        Initialise the aggregator
        """
        self.logger = logger
        #*** Retrieve parameters passed to us:
        self.datasets = datasets
        self.parameters = parameters
        self.experiment_name = parameters['experiment_name']
        self.dataset_name = parameters['dataset']
        self.dataset = datasets[parameters['dataset']]
        self.iterations = parameters['iterations']
        self.partions_number = parameters['partions_number']
        self.alg = parameters['alg']
        self.experiment_policy = parameters['experiment_policy']
        self.result_range = parameters['result_range']
        self.result_threshold = parameters['result_threshold']

    def run(self):
        """
        Run cross validation of the experiment
        """
        self.logger.debug("running cross validation of experiment name=%s with"
                                " %s partitions", self.experiment_name,
                                self.partions_number)
        results = self._cross_validate()
        self.logger.debug("results are %s", results)
        return results

    def _cross_validate(self):
        """
        Perform a cross validation test on a given algorithm with
        a given dataset and number of partitions (folds) to use
        """
        #*** Run cross validation by setting each partition in turn
        #*** to be validation with all others as training:
        results_set = []
        alg_parameters = {'partition': 'Training',
                      'dataset': self.dataset_name,
                      'iterations': self.iterations}
        for index in xrange(self.partions_number):
            #*** Reinitialise the algorithm:
            self.alg.initialise()

            #*** Set partitions:
            partitions_list = ['Training'] * self.partions_number
            partitions_list[index] = 'Validation'
            self.logger.debug("Setting partitions to %s", partitions_list)
            self.dataset.partition(partitions_list)

            #*** Run training:
            self.alg.train(self.datasets, alg_parameters)

            #*** Run test:
            results = self.alg.test(self.datasets, alg_parameters)
            self.logger.info("results=%s", results)
            results_accuracy= self.result_analysis(results,
                                                         self.result_threshold)
            results_set.append(results_accuracy)
        return results_set

    def result_analysis(self, results, threshold):
        """
        Apply post processing analysis to raw results
        and return an integer between 0 and 100 that indicates accuracy
        percentage
        """
        correct = 0
        incorrect = 0
        for result in results:
            actual = result['actual']
            computed = result['computed']
            if actual > computed:
                variance = actual - computed
            else:
                variance = computed - actual
            if variance < threshold:
                correct += 1
            else:
                incorrect += 1
        #*** Sanity check:
        assert correct + incorrect == len(results)
        #*** Calculate percentage accuracy:
        if len(results):
            accuracy_percent = (correct / len(results)) * 100
        else:
            accuracy_percent = 0
        self.logger.info("correct=%s, incorrect=%s, accuracy_percent=%s",
                                          correct, incorrect, accuracy_percent)
        return accuracy_percent
    
