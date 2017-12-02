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

Evaluate library provides a class with methods to evaluate result
data against desired results
"""

from __future__ import division

class Evaluate(object):
    """
    Class with methods for evaluating result data
    """
    def __init__(self, logger):
        """
        Initialise the evaluation
        """
        self.logger = logger

    def simple_accuracy(self, results, threshold):
        """
        Evaluation of simple results data that is in the form of a
        list of dictionaries, each of which contain two KVPs:
        
        * actual
        * computed
        
        All result values are floats
        
        A threshold is passed in, and if the actual result is +/-
        threshold of computed result then it is recorded as correct
        otherwise incorrect.
        
        Returns accuracy percentage as an integer between 0 and 100.
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
            if variance <= threshold:
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
        self.logger.info("EVALUATE: correct=%s, incorrect=%s, accuracy_percent=%s",
                                          correct, incorrect, accuracy_percent)
        return accuracy_percent


