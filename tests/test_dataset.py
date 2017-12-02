"""
AMLE dataset.py Unit Tests
"""

#*** Handle tests being in different directory branch to app code:
import sys
sys.path.insert(0, '../amle')

#*** Testing imports:
import unittest

#*** Logging:
import logging
import coloredlogs

#*** Ordered Dictionaries:
from collections import OrderedDict

#*** numpy for mathematical functions:
from numpy import array_equal

#*** AMLE imports:
import dataset as dataset_module

#*** Set up logging:
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,
                    fmt="%(asctime)s %(module)s[%(process)d] %(funcName)s " + 
                    "%(levelname)s %(message)s",
                    datefmt='%H:%M:%S')

def test_trim_to_rows():
    """
    Test the trim_to_rows method
    """
    dset = dataset_module.DataSet(logger)
    dset.ingest('data/test/test1.csv')
    dset.trim_to_rows('dave', ['foo', 'fighter'])
    logger.info("get_data=%s", dset.get_data())
    assert dset.get_data() == [OrderedDict([('alice', '1'), ('bob', '1'), ('charlie', '10'), ('dave', 'foo')]),
                               OrderedDict([('alice', '3'), ('bob', '4'), ('charlie', '50'), ('dave', 'fighter')])]

def test_trim_to_columns():
    """
    Test the trim_to_columns method
    """
    dset = dataset_module.DataSet(logger)
    dset.ingest('data/test/test1.csv')
    dset.trim_to_columns(['alice', 'charlie'])
    logger.info("get_data=%s", dset.get_data())
    assert dset.get_data() == [OrderedDict([('alice', '1'), ('charlie', '10')]),
                               OrderedDict([('alice', '2'), ('charlie', '20')]),
                               OrderedDict([('alice', '3'), ('charlie', '50')])]

def test_duplicate_column():
    """
    Test the duplicate_column method
    """
    dset = dataset_module.DataSet(logger)
    dset.ingest('data/test/test1.csv')
    dset.duplicate_column('alice', 'eve')
    logger.info("get_data=%s", dset.get_data())
    assert dset.get_data() == [OrderedDict([('alice', '1'), ('bob', '1'), ('charlie', '10'), ('dave', 'foo'), ('eve', '1')]),
                               OrderedDict([('alice', '2'), ('bob', '2'), ('charlie', '20'), ('dave', 'bar'), ('eve', '2')]),
                               OrderedDict([('alice', '3'), ('bob', '4'), ('charlie', '50'), ('dave', 'fighter'), ('eve', '3')])]

def test_rescale():
    """
    Test the rescale method
    """
    dset = dataset_module.DataSet(logger)
    dset.ingest('data/test/test1.csv')
    dset.rescale('charlie', 10, 50)
    logger.info("get_data=%s", dset.get_data())
    assert dset.get_data() == [OrderedDict([('alice', '1'), ('bob', '1'), ('charlie', 0.0), ('dave', 'foo')]),
                               OrderedDict([('alice', '2'), ('bob', '2'), ('charlie', 0.25), ('dave', 'bar')]),
                               OrderedDict([('alice', '3'), ('bob', '4'), ('charlie', 1.0), ('dave', 'fighter')])]

def test_translate():
    """
    Test the translate method
    """
    dset = dataset_module.DataSet(logger)
    dset.ingest('data/test/test1.csv')
    dset.translate('dave', {'foo': 'oof', 'bar': 1, 'fighter': 0})
    logger.info("get_data=%s", dset.get_data())
    assert dset.get_data() == [OrderedDict([('alice', '1'), ('bob', '1'), ('charlie', '10'), ('dave', 'oof')]),
                               OrderedDict([('alice', '2'), ('bob', '2'), ('charlie', '20'), ('dave', 1)]),
                               OrderedDict([('alice', '3'), ('bob', '4'), ('charlie', '50'), ('dave', 0)])]

def test_one_hot_encode():
    """
    Test the one_hot_encode method
    """
    #*** Test standard one hot encoding but with one match value set to two instead of one:
    dset = dataset_module.DataSet(logger)
    dset.ingest('data/test/test1.csv')
    key_values = OrderedDict([('foo', 1), ('bar', 1), ('fighter', 2)])
    dset.one_hot_encode('dave', key_values)
    logger.info("get_data=%s", dset.get_data())
    assert dset.get_data() == [OrderedDict([('alice', '1'), ('bob', '1'), ('charlie', '10'), ('dave', 'foo'), ('foo', 1), ('bar', 0), ('fighter', 0)]),
                               OrderedDict([('alice', '2'), ('bob', '2'), ('charlie', '20'), ('dave', 'bar'), ('foo', 0), ('bar', 1), ('fighter', 0)]),
                               OrderedDict([('alice', '3'), ('bob', '4'), ('charlie', '50'), ('dave', 'fighter'), ('foo', 0), ('bar', 0), ('fighter', 2)])]

class TestConfig(unittest.TestCase):
    def test_one_hot_encode2(self):
        #*** Test column name collision protection:
        dset = dataset_module.DataSet(logger)
        dset.ingest('data/test/test1.csv')
        dset.translate('dave', {'foo': 'alice'})
        key_values = OrderedDict([('alice', 1), ('bar', 1), ('fighter', 2)])
        with self.assertRaises(SystemExit):
            dset.one_hot_encode('dave', key_values)

def test_shuffle():
    """
    Test the shuffle method, setting random seed for predictable test result
    """
    dset = dataset_module.DataSet(logger)
    dset.ingest('data/test/test1.csv')
    dset.shuffle(seed=1)
    logger.info("get_data=%s", dset.get_data())
    assert dset.get_data() == [OrderedDict([('alice', '2'), ('bob', '2'), ('charlie', '20'), ('dave', 'bar')]),
                               OrderedDict([('alice', '3'), ('bob', '4'), ('charlie', '50'), ('dave', 'fighter')]),
                               OrderedDict([('alice', '1'), ('bob', '1'), ('charlie', '10'), ('dave', 'foo')])]

def test_partition():
    """
    Test the partition method and ensure it produces
    anticipated results when dataset outputs are requested
    """
    #===================================
    #*** TEST 1:
    dset = dataset_module.DataSet(logger)
    dset.ingest('data/test/test1.csv')
    dset.translate('dave', {'foo': 0, 'bar': 0.5, 'fighter': 1})
    dset.partition(['Training', 'Validation'])
    
    #** Comparison that is trimmed to rows to equal partition result:
    dset2 = dataset_module.DataSet(logger)
    dset2.ingest('data/test/test1.csv')
    dset2.trim_to_rows('dave', ['foo', 'fighter'])
    dset2.translate('dave', {'foo': 0, 'bar': 0.5, 'fighter': 1})

    logger.info("dset  inputs_array=%s", dset.inputs_array(partition='Training'))
    logger.info("dset2 inputs_array=%s", dset2.inputs_array())

    #*** Use a numpy function to compare arrays:
    assert array_equal(dset.inputs_array(partition='Training'), dset2.inputs_array())

    #===================================
    #*** TEST 2:
    dset = dataset_module.DataSet(logger)
    dset.ingest('data/test/test1.csv')
    dset.translate('dave', {'foo': 0, 'bar': 0.5, 'fighter': 1})
    dset.partition(['Training', 'Training', 'Training', 'Validation'])
    
    #** Comparison array:
    dset2 = dataset_module.DataSet(logger)
    dset2.ingest('data/test/test1.csv')
    dset2.translate('dave', {'foo': 0, 'bar': 0.5, 'fighter': 1})

    logger.info("dset  inputs_array=%s", dset.inputs_array(partition='Training'))
    logger.info("dset2 inputs_array=%s", dset2.inputs_array())

    #*** Use a numpy function to compare arrays:
    assert array_equal(dset.inputs_array(partition='Training'), dset2.inputs_array())

    #===================================
    #*** TEST 3 (outputs_array test):
    dset = dataset_module.DataSet(logger)
    dset.ingest('data/test/test1.csv')
    dset.translate('dave', {'foo': 0, 'bar': 0.5, 'fighter': 1})
    dset.set_output_columns(['bob', 'charlie'])
    dset.partition(['Training', 'Training', 'Validation'])
    
    #** Comparison array:
    dset2 = dataset_module.DataSet(logger)
    dset2.ingest('data/test/test1.csv')
    dset2.trim_to_rows('dave', ['foo', 'bar'])
    dset2.translate('dave', {'foo': 0, 'bar': 0.5, 'fighter': 1})
    dset2.set_output_columns(['bob', 'charlie'])

    logger.info("dset  outputs_array=%s", dset.outputs_array(partition='Training'))
    logger.info("dset2 outputs_array=%s", dset2.outputs_array())

    #*** Use a numpy function to compare arrays:
    assert array_equal(dset.outputs_array(partition='Training'), dset2.outputs_array())

def test_in_partition():
    """
    Test the in_partition method
    """
    dset = dataset_module.DataSet(logger)
    dset.partition(partitions=["one", "two", "three", "four", "five"])
    assert dset.in_partition('one', 0) == 1
    assert dset.in_partition('two', 1) == 1
    assert dset.in_partition('three', 2) == 1
    assert dset.in_partition('four', 3) == 1
    assert dset.in_partition('five', 4) == 1
    assert dset.in_partition('one', 5) == 1
    assert dset.in_partition('two', 6) == 1
    assert dset.in_partition('three', 7) == 1
    assert dset.in_partition('four', 8) == 1
    assert dset.in_partition('five', 9) == 1
    #*** Counter cases:
    assert dset.in_partition('one', 21) == 0
    assert dset.in_partition('two', 22) == 0
    assert dset.in_partition('three', 23) == 0
    assert dset.in_partition('four', 24) == 0
    assert dset.in_partition('five', 25) == 0



        
