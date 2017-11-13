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

#*** CSV library:
import csv

#*** Ordered Dictionaries:
from collections import OrderedDict

#*** numpy for ML:
from numpy import exp, asarray, array, random, dot, matrix, float32

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
        #*** List of dictionaries (rows) that holds the data:
        self.data = []
        #*** Subset of data that contains column names for output data:
        self.output_columns = []

    def set_output_columns(self, output_columns):
        """
        Set what columns are used as output data from dataset
        (i.e. what columns contain the expected answer(s)
        Pass it a list of output column names
        """
        self.output_columns = output_columns

    def display(self):
        """
        Print data to screen
        """
        for row in self.data:
            print row

    def ingest(self, filename):
        """
        Load data CSV from file into class as
        a list of dictionaries of rows. Requires first row in
        file to be a header row and uses these values as keys
        in row dictionaries. Example row:
        {'dataset': 'ML', 'min_interpacket_interval': '0.001'}
        """
        self.data = []
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
                self.data.append(sorted_row)

    def translate(self, column_name, value_original, value_xlate):
        """
        Go through all values in a column replacing any occurences
        of value_original with value_xlate
        """
        result = []
        for row in self.data:
            if row[column_name] == value_original:
                row[column_name] = value_xlate
            result.append(row)
        self.data = result

    def trim_to_columns(self, fields):
        """
        Passed a list of fields (columns) to
        retain and trim the internal representation of the training
        data to just those columns
        """
        result = []
        for row in self.data:
            row_result = OrderedDict()
            for row_item_key in row:
                if row_item_key in fields:
                    row_result[row_item_key] = row[row_item_key]
            result.append(row_result)
        self.data = result

    def trim_to_rows(self, key, fields):
        """
        Passed a key (column name) and list of fields (column values)
        match rows that should be retained and remove other rows
        """
        result = []
        for row in self.data:
            for field in fields:
                if field == row[key]:
                    result.append(row)
        self.data = result

    def inputs_array(self):
        """
        Return input data as a numpy array
        Filter out output column(s)
        """
        #*** Create a subset without the output column(s):
        data_input_subset = []
        for row in self.data:
            row_result = OrderedDict()
            for row_item_key in row:
                if row_item_key not in self.output_columns:
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
        for row in self.data:
            row_result = OrderedDict()
            for row_item_key in row:
                if row_item_key in self.output_columns:
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
        result = []
        for row in self.data:
            row[column_name] = \
                            (float(row[column_name]) - min_x) / (max_x - min_x)
            result.append(row)
        self.data = result
