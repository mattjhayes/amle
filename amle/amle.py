#!/usr/bin/python

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Automated Machine Learning Environment (AMLE)

This code is a simple shim to help automate running
machine learning (ML) tests to reduce effort and
make it easier to innovate.

Requires various packages including YAML:
    sudo apt-get install python-pip git git-flow python-pytest python-yaml

Requires PIP packages coloredlogs, voluptuous and numpy. Install with:
    pip install coloredlogs voluptuous numpy

Principles (aspirational):

* Generic. Just a shim, does not contain ML code, and tries
  to not be opinionated about how ML works or data types
* Reproducibility. Run the same test with same inputs and
  get the same output(s) - or at least statistically similar.
* Reduce experimentation work effort. Support comparative
  testing across different parameters and/or ML algorithms,
  retains historical parameters and results
* Add value to experimentation. Support evolutionary genetic
  approach to configuring algorithm parameters
* Visibility. Make it easy to understand how experiments are
  running / ran
"""

#*** For file path:
import os

#*** Import sys and getopt for command line argument parsing:
import sys
import getopt

#*** Logging:
import logging

import traceback

#*** For dynamic importing of modules:
import importlib

#*** Colorise the logs:
import coloredlogs

#*** AMLE project imports:
import config
import policy
import dataset
#*** AMLE, for logging configuration:
from baseclass import BaseClass

VERSION = "0.1.0"

#*** Configure Logging:
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,
                    fmt="%(asctime)s %(module)s[%(process)d] %(funcName)s " +
                    "%(levelname)s %(message)s",
                    datefmt='%H:%M:%S')

class AMLE(BaseClass):
    """
    This class provides core methods for an Automated Machine Learning
    Environment (AMLE)
    """
    def __init__(self, CLI_arguments):
        """
        Initialise the AMLE class
        """
        #*** Instantiate config class which imports configuration file
        #*** config.yaml and provides access to keys/values:
        self.config = config.Config()

        #*** Now set config module to log properly:
        self.config.inherit_logging(self.config)

        #*** Set up Logging with inherited base class method:
        self.configure_logging(__name__, "amle_logging_level_s",
                                       "amle_logging_level_c")

        #*** Update sys.path (PYTHONPATH) for loading custom classifiers:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        self.logger.debug("sys.path=%s", sys.path)

        #*** Parse command line parameters:
        try:
            opts, args = getopt.getopt(CLI_arguments, "hp:v",
                                    ["help",
                                    "project=",
                                    "version"])
        except getopt.GetoptError as err:
            logger.critical('AMLE: Error with options: %s', err)
            print_help()
            sys.exit(2)
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print_help()
                sys.exit()
            elif opt in ("-v", "--version"):
                print "\n\n AMLE version", VERSION, "\n"
                sys.exit()
            elif opt in ("-p", "--project"):
                self.project_directory = arg

        #*** Must have a policy file specified:
        if not self.project_directory:
            logger.critical("No project directory specified, exiting")
            sys.exit()

        #*** Instantiate Module Classes:
        self.policy = policy.Policy(self.config, self.project_directory)

        #*** Dictionary to hold dataset objects:
        self._datasets = {}
        #*** Dictionary to hold algorithm objects:
        self._algorithms = {}
        #*** Dictionary to hold aggregator objects:
        self._aggregators = {}

    def run(self):
        """
        Run AMLE
        """
        #*** Load and pre-process datasets:
        policy_datasets = self.policy.get_datasets()
        for policy_dataset in policy_datasets:
            #*** Create dataset object and ingest data from file:
            dset = dataset.DataSet(logger)
            dset.set_name(policy_dataset['name'])
            dset.ingest(policy_dataset['source']['file'])
            #*** Run transforms to process dataset into right form:
            dset.transform(policy_dataset['transform'])
            #*** Add dataset to datasets dictionary:
            self._datasets[policy_dataset['name']] = dset

        #*** Load algorithms:
        policy_algorithms = self.policy.get_algorithms()
        for policy_algorithm in policy_algorithms:
            alg = self.load_algorithm(policy_algorithm['location'])
            self._algorithms[policy_algorithm['name']] = alg

        #*** Load aggregators (optional):
        policy_aggregators = self.policy.get_aggregators()
        for policy_aggregator in policy_aggregators:
            agg = self.load_aggregator(policy_aggregator['location'])
            self._aggregators[policy_aggregator['name']] = agg

        #*** Run section:
        run_items = self.policy.get_run_items()
        for run_item in run_items:
            #*** Run the item:
            self.logger.debug("running item=%s", run_item['name'])
            if 'aggregator' in run_item:
                name = run_item['aggregator']['name']
                parameters = run_item['aggregator']['parameters']
                self.run_aggregator(name, parameters)
            else:
                #*** No aggregator so run experiment direct:
                self.run_experiment(run_item['experiment'])

    def run_experiment(self, experiment_name):
        """
        Run an experiment, as per spec from policy
        """
        self.logger.debug("running experiment=%s", experiment_name)
        pol_exp = self.policy.get_experiment(experiment_name)
        algr_name = pol_exp['algorithm']['name']
        algr_parameters = pol_exp['algorithm']['parameters']
        self.logger.debug("Initiating algorithm=%s with parameters=%s",
                                    algr_name, algr_parameters)
        algr = self._algorithms[algr_name](self.logger, algr_parameters)
        #*** Run specified methods in experiment class:
        methods = pol_exp['algorithm']['methods']
        for method_dict in methods:
            self.logger.debug("method_dict=%s", method_dict)
            method = next(iter(method_dict))
            self.logger.debug("method=%s", method)
            parameters = method_dict[method]['parameters']
            self.logger.debug("parameters=%s", parameters)
            result = getattr(algr, method)(self._datasets, parameters)
            self.logger.debug("result=%s", result)

    def run_aggregator(self, agg_name, agg_parameters):
        """
        Run an aggregator, as per spec from policy
        """
        self.logger.debug("Preparing to run aggregator=%s", agg_name)
        pol_agg = self.policy.get_aggregator(agg_name)
        self.logger.debug("pol_agg=%s", pol_agg)
        #*** Start experiment:
        name = agg_parameters['experiment']
        #      dataset: training_dataset
        #      iterations: 60000
        #      partions_number: 4
        self.logger.debug("Initialising algorithm for name=%s", name)
        pol_experiment = self.policy.get_experiment(name)
        alg_name = pol_experiment['algorithm']['name']
        alg_parameters = pol_experiment['algorithm']['parameters']
        self.logger.debug("Initiating algorithm=%s with parameters=%s",
                                                      alg_name, alg_parameters)
        alg = self._algorithms[alg_name](self.logger, alg_parameters)
        agg_parameters['experiment_name'] = name
        agg_parameters['alg'] = alg
        agg_parameters['experiment_policy'] = pol_experiment

        #*** Start aggregator:
        self.logger.debug("Starting aggregator=%s with parameters=%s",
                                                      agg_name, agg_parameters)
        agg = self._aggregators[agg_name](self.logger, self._datasets,
                                                                agg_parameters)
        result = getattr(agg, 'run')()
        self.logger.debug("result=%s", result)

    def load_algorithm(self, alg_name):
        """
        Passed file location for an algorithm
        module and return it as an object
        """
        self.logger.debug("Loading algorithm=%s", alg_name)
        #*** Replace directory forward slashes with dots, Unix-specific:
        alg_name = alg_name.replace("/", ".")
        self.logger.debug("module=%s", alg_name)
        #*** Try importing the module:
        try:
            module = importlib.import_module(alg_name)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.logger.error("Failed to dynamically load "
                                "module=%s\nPlease check that module exists "
                                "and alter project_policy configuration "
                                "if required",
                                alg_name)
            self.logger.error("Exception is %s, %s, %s",
                                            exc_type, exc_value,
                                            traceback.format_tb(exc_traceback))
            sys.exit("Exiting, please fix error...")

        #*** Dynamically instantiate class 'Classifier':
        self.logger.debug("Instantiating module class alg_name=%s", alg_name)
        class_ = getattr(module, 'Algorithm')
        return class_

    def load_aggregator(self, agg_name):
        """
        Passed file location for an aggregator
        module and return it as an object
        """
        self.logger.debug("Loading aggregator=%s", agg_name)
        #*** Replace directory forward slashes with dots, Unix-specific:
        agg_name = agg_name.replace("/", ".")
        self.logger.debug("module=%s", agg_name)
        #*** Try importing the module:
        try:
            module = importlib.import_module(agg_name)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.logger.error("Failed to dynamically load "
                                "module=%s\nPlease check that module exists "
                                "and alter project_policy configuration "
                                "if required",
                                agg_name)
            self.logger.error("Exception is %s, %s, %s",
                                            exc_type, exc_value,
                                            traceback.format_tb(exc_traceback))
            sys.exit("Exiting, please fix error...")

        #*** Dynamically instantiate class 'Aggregator':
        self.logger.debug("Instantiating module class agg_name=%s", agg_name)
        class_ = getattr(module, 'Aggregator')
        return class_

def print_help():
    """
    Print out the help instructions
    """
    print """
Automated Machine Learning Environment (AMLE)
---------------------------------------------

Use this very basic shim to train and evaluate the
performance of machine learning classifiers.

Usage:
  python amle.py --project PROJECT_DIRECTORY [options]

Example usage:
  python amle.py --project projects/examples/project1

Options:
 -h  --help          Display this help and exit
 -p  --project       Specify a project directory, relative to amle root
                     (mandatory)
 -v  --version       Output version information and exit

 """
    return()

if __name__ == "__main__":
    #*** Instantiate the AMLE class:
    amle = AMLE(sys.argv[1:])
    #*** Start AMLE with command line arguments from position 1:
    amle.run()

