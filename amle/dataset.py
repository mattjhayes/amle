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
The dataset module provides an abstraction for
sets of data, primarily aimed at use in machine
learning (ML).
"""

import os
import sys
import random

#*** CSV library:
import csv

#*** Ordered Dictionaries:
from collections import OrderedDict

#*** numpy for mathematical functions:
from numpy import array

class DataSet(object):
    """
    Represents a set of ML data with methods to ingest,
    manipulate (i.e. preprocess) and extract
    """
    def __init__(self, logger):
        """
        Initialise the class
        """
        self.logger = logger
        #*** Name for the dataset:
        self._name = ""
        #*** List of dictionaries (rows) that holds the data:
        self._data = []
        #*** Subset of data that contains column names for output data:
        self._output_columns = []
        #*** Default partition configuration:
        self._divisor = 1
        self._partitions = ['A']

    def get_data(self):
        """
        Return data in native format
        """
        return self._data

    def set_output_columns(self, output_columns):
        """
        Set what columns are used as output data from dataset
        (i.e. what columns contain the expected answer(s)
        Pass it a list of output column names
        """
        self.logger.debug("Setting output_columns=%s", output_columns)
        self._output_columns = output_columns

    def set_name(self, name):
        """
        Set the name for the dataset
        """
        self._name = name

    def display(self, display_type):
        """
        Display data
        """
        if display_type == 'print':
            print "\nDataset: " + self._name + "\n"
            for row in self._data:
                print row
        else:
            self.logger.critical("Unsupported display_type=%s, exiting...",
                                                                  display_type)
            sys.exit()

    def ingest(self, filename):
        """
        Load data CSV from file into class as
        a list of dictionaries of rows. Requires first row in
        file to be a header row and uses these values as keys
        in row dictionaries. Example row:
        {'dataset': 'ML', 'min_interpacket_interval': '0.001'}
        """
        self._data = []
        working_directory = os.path.dirname(__file__)
        fullpathname = os.path.join(working_directory, filename)
        if os.path.isfile(fullpathname):
            self.logger.info("Ingesting file=%s", fullpathname)
        else:
            self.logger.critical("Dataset=%s not found, exiting", fullpathname)
            sys.exit()
        with open(fullpathname) as filehandle:
            reader = csv.DictReader(filehandle)
            for row in reader:
                sorted_row = OrderedDict(sorted(row.items(),
                            key=lambda item: reader.fieldnames.index(item[0])))
                self._data.append(sorted_row)

    def transform(self, transform_policy):
        """
        Passed policy transforms and run them against the dataset.
        """
        self.logger.debug("Running transforms on dataset")
        for tform in transform_policy:
            self.logger.debug("transform is %s", tform)
            if 'trim_to_rows' in tform:
                for row in tform['trim_to_rows']:
                    for key in row:
                        self.trim_to_rows(key, row[key])
            elif 'trim_to_columns' in tform:
                self.trim_to_columns(tform['trim_to_columns'])
            elif 'rescale' in tform:
                rdict = tform['rescale'][0]
                self.rescale(rdict['column'], rdict['min'], rdict['max'])
            elif 'translate' in tform:
                rlist = tform['translate']
                self.translate(rlist[0]['column'], rlist[1]['values'])
            elif 'set_output_columns' in tform:
                self.set_output_columns(tform['set_output_columns'])
            elif 'shuffle' in tform:
                self.shuffle(seed=tform['shuffle'])
            elif 'display' in tform:
                self.display(tform['display'])
            else:
                self.logger.critical("Unsupported transform=%s, exiting...",
                                                                         tform)
                sys.exit()

    def shuffle(self, seed=0):
        """
        Shuffle dataset rows.
        Set seed=1 if want predictable randomness for reproduceable
        shuffling
        """
        if seed:
            random.Random(4).shuffle(self._data)
        else:
            random.shuffle(self._data)

    def partition(self, divisor=1, partitions=['A']):
        """
        Set partition parameters for split of dataset into
        arbitrary partitions, which are named by strings.
        Note that partitioning is applied when data is retrieved,
        not to internal dataset
        - divisor is how many sets to divide data into
        - partitions is allocation of the sets to named partitions

        Setting partition values overwrites any previously set
        partition configuration

        Default partition is divisor=1, partitions=['A']
        (i.e. all data in partition 'A')

        Standard convention for usage of partitions is:
        * Partition 'Training' is used as training data
        * Partition 'Validation' is used as validation (test) data

        Example: Randomise row order, then allocate 75% of rows to
        partition 'Training' with the last 25% in partition 'Validation':
          dataset.shuffle()
          dataset.partition(divisor=4, partitions=['Training',
                        'Training', 'Training', 'Validation'])
        """
        #*** Sanity check:
        if len(partitions) != divisor:
            self.logger.critical("Partitions allocation list=%s length not "
                                "equal to divisor=%s, exiting...", partitions,
                                divisor)
            sys.exit()
        self._divisor = divisor
        self._partitions = partitions

    def in_partition(self, partition_name, row_number):
        """
        Passed a partition name, row number and total number of
        rows in the dataset and after consulting internal
        partition settings, return a 1 if the given row
        belongs to the partition, otherwise 0
        """
        remainder = row_number % self._divisor
        return self._partitions[remainder] == partition_name

    def translate(self, column_name, value_mapping):
        """
        Go through all values in a column replacing any occurences
        of key in value_mapping dictionary with corresponding value
        """
        self.logger.debug("Translating column_name=%s values=%s",
                                                    column_name, value_mapping)
        result = []
        for row in self._data:
            if row[column_name] in value_mapping:
                row[column_name] = value_mapping[row[column_name]]
            result.append(row)
        self._data = result

    def trim_to_columns(self, fields):
        """
        Passed a list of fields (columns) to
        retain and trim the internal representation of the training
        data to just those columns
        """
        self.logger.debug("Trimming dataset to only column_name=%s", fields)
        result = []
        for row in self._data:
            row_result = OrderedDict()
            for row_item_key in row:
                if row_item_key in fields:
                    row_result[row_item_key] = row[row_item_key]
            result.append(row_result)
        self._data = result

    def trim_to_rows(self, key, fields):
        """
        Passed a key (column name) and list of fields (column values)
        match rows that should be retained and remove other rows
        """
        self.logger.debug("Trimming dataset to where column_name=%s value=%s",
                                                                   key, fields)
        result = []
        for row in self._data:
            for field in fields:
                if field == row[key]:
                    result.append(row)
        self._data = result

    def inputs_array(self, partition='A'):
        """
        Return input data as a numpy array
        Filter out output column(s) and only include
        rows from specified partition, which defaults
        to 'A'
        """
        #*** Create a subset without the output column(s):
        data_input_subset = []
        for index, row in enumerate(self._data):
            if self.in_partition(partition, index):
                row_result = OrderedDict()
                for row_item_key in row:
                    if row_item_key not in self._output_columns:
                        row_result[row_item_key] = row[row_item_key]
                data_input_subset.append(row_result)
        #*** Now convert into numpy array:
        list_of_lists = []
        for row in data_input_subset:
            row_values = row.values()
            row = [float(x) for x in row_values]
            list_of_lists.append(row)
        return array(list_of_lists)

    def outputs_array(self):
        """
        Return output data as a numpy array
        Filter out input columns
        """
        #*** Create a subset without the input column(s):
        data_output_subset = []
        for row in self._data:
            row_result = OrderedDict()
            for row_item_key in row:
                if row_item_key in self._output_columns:
                    row_result[row_item_key] = row[row_item_key]
            data_output_subset.append(row_result)
        #*** Now convert into numpy array:
        list_of_lists = []
        for row in data_output_subset:
            row_values = row.values()
            row = [float(x) for x in row_values]
            list_of_lists.append(row)
        return array(list_of_lists)

    def rescale(self, column_name, min_x, max_x):
        """
        Rescale all values in a column so that they sit between
        0 and 1. Uses rescaling formula:
        x` = (x - min(x)) / (max(x) - min(x))
        """
        self.logger.debug("Rescaling dataset column_name=%s min=%s max=%s",
                                                     column_name, min_x, max_x)
        result = []
        for row in self._data:
            row[column_name] = \
                            (float(row[column_name]) - min_x) / (max_x - min_x)
            result.append(row)
        self._data = result
