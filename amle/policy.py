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

Policy library that handles reading in policy,
validating it and providing values to other
parts of AMLE
"""

import sys
import os
import datetime

#*** Voluptuous to verify inputs against schema:
from voluptuous import Schema, Optional, Any, All, Required, Extra
from voluptuous import Invalid, MultipleInvalid, Range

#*** YAML for config and policy file parsing:
import yaml

#*** Regular Expressions:
import re

#*** For logging configuration:
from baseclass import BaseClass

POLICY_FILENAME = 'project_policy.yaml'

#================== Functions (need to come first):

def validate(logger, data, schema, where):
    """
    Generic validation of a data structure against schema
    using Voluptuous data validation library
    Parameters:
     - logger: valid logger reference
     - data: structure to validate
     - schema: a valid Voluptuous schema
     - where: string for debugging purposes to identity the policy location
    """
    logger.debug("validating data=%s", data)
    try:
        #*** Check correctness of data against schema with Voluptuous:
        schema(data)
    except MultipleInvalid as exc:
        #*** There was a problem with the data:
        logger.critical("Policy syntax problem where=%s data=%s exception=%s "
                    "error=%s", where, yaml.dump(data), exc, exc.errors)
        sys.exit("Exiting AMLE. Please fix error in " + POLICY_FILENAME) 
    return 1

#================= Voluptuous Schema for Validating Policy

#*** Voluptuous schema keys / value types in the policy:
TOP_LEVEL_SCHEMA = Schema({
                        Required('datasets'): list,
                        Required('algorithms'): list,
                        Required('experiments'): list
                        })
DATASET_SCHEMA = Schema({
                        Required('name'): str,
                        Required('source'): dict,
                        Required('transform'): list
                        })
TRANSFORM_SCHEMA = Schema([{
                        Optional('trim_to_rows'): list,
                        Optional('trim_to_columns'): list,
                        Optional('rescale'): list,
                        Optional('translate'): list,
                        Optional('set_output_columns'): list,
                        Optional('display'): str
                        }
                        ])


class Policy(BaseClass):
    """
    This policy class serves these purposes:
    - Ingest policy (policy.yaml) from file
    - Validate correctness of policy against schema
    - Methods and functions to check various parameters
      against policy
    Note: Class definitions are not nested as not considered Pythonic
    Main Methods and Variables:
    - ingest    # Read in policy and check validity
    """
    def __init__(self, config, project_directory):
        """ Initialise the Policy Class """
        #*** Required for BaseClass:
        self.config = config
        #*** Set up Logging with inherited base class method:
        self.configure_logging(__name__, "policy_logging_level_s",
                                       "policy_logging_level_c")
        logger = self.logger
        self.project_directory = project_directory
        #*** Get working directory:
        self.working_directory = os.path.dirname(__file__)
        #*** Build the full path and filename for the project policy file:
        self.fullpathname = os.path.join(self.working_directory,
                                         self.project_directory,
                                         POLICY_FILENAME)
        if os.path.isfile(self.fullpathname):
            logger.info("Opening project policy file=%s", self.fullpathname)
        else:
            logger.critical("Project policy file=%s not found, exiting",
                                                            self.fullpathname)
            sys.exit()
        #*** Ingest the policy file:
        try:
            with open(self.fullpathname, 'r') as filename:
                self.policy = yaml.safe_load(filename)
        except (IOError, OSError) as exception:
            logger.error("Failed to open policy file=%s exception=%s",
                                                  self.fullpathname, exception)
            sys.exit("Exiting AMLE. Please create " + POLICY_FILENAME) 
        #*** Check the correctness of the policy:
        validate(logger, self.policy, TOP_LEVEL_SCHEMA, 'top')
        for dataset in self.policy['datasets']:
            validate(logger, dataset, DATASET_SCHEMA, 'dataset')
            validate(logger, dataset['transform'], TRANSFORM_SCHEMA,
                                                                   'transform')
        logger.warning("TBD: need to finish policy validation...")

    def get_datasets(self):
        """
        Return a list of policy datasets
        """
        result = []
        for dataset in self.policy['datasets']:
            result.append(dataset)
        return result

    def get_algorithms(self):
        """
        Return a list of policy algorithms
        """
        result = []
        for algorithm in self.policy['algorithms']:
            result.append(algorithm)
        return result
