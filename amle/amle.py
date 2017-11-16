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
  running
"""

VERSION = "0.1.0"

#*** For file path:
import os

#*** Import sys and getopt for command line argument parsing:
import sys, getopt

#*** Logging:
import logging
import coloredlogs

import traceback

#*** For dynamic importing of modules:
import importlib

#*** AMLE project imports:
import config
import policy
import dataset
#*** AMLE, for logging configuration:
from baseclass import BaseClass

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

    def run(self):
        """
        Run AMLE
        """
        #*** Create dataset objects based on policy directives:
        policy_datasets = self.policy.get_datasets()
        for policy_dataset in policy_datasets:
            #*** Create dataset object and ingest data from file:
            dset = dataset.DataSet(logger)
            dset.set_name(policy_dataset['name'])
            dset.ingest(policy_dataset['source']['file'])
            #*** Run transforms to process dataset into right form:
            self.transform(dset, policy_dataset['transform'])
            #*** Add dataset to datasets dictionary:
            self._datasets[policy_dataset['name']] = dset
        #*** Load algorithms:
        policy_algorithms = self.policy.get_algorithms()
        for policy_algorithm in policy_algorithms:
            alg = self.load_algorithm(policy_algorithm['code'])
            self._algorithms[policy_algorithm['name']] = alg

        #*** Now run experiments:
        
        # TEMP:
        dset = self._datasets['training_dataset']
        algr = self._algorithms['ANN_simple_2_layer'](self.logger)
        algr.run(dset.inputs_array(), dset.outputs_array())

    def transform(self, dset, transform_policy):
        """
        Passed a dataset object and relevant transforms to run on it.
        Run the transforms in the dataset object
        """
        self.logger.debug("Running transforms on dataset")
        for tform in transform_policy:
            self.logger.debug("transform is %s", tform)
            if 'trim_to_rows' in tform:
                for row in tform['trim_to_rows']:
                    for key in row:
                        dset.trim_to_rows(key, row[key])
            elif 'trim_to_columns' in tform:
                dset.trim_to_columns(tform['trim_to_columns'])
            elif 'rescale' in tform:
                rdict = tform['rescale'][0]
                dset.rescale(rdict['column'], rdict['min'], rdict['max'])
            elif 'translate' in tform:
                rlist = tform['translate']
                dset.translate(rlist[0]['column'], rlist[1]['values'])
            elif 'set_output_columns' in tform:
                dset.set_output_columns(tform['set_output_columns'])
            elif 'display' in tform:
                dset.display(tform['display'])
            else:
                self.logger.critical("Unsupported transform=%s, exiting...",
                                                                         tform)
                sys.exit()

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
                                "module=%s .Please check that module exists "
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

