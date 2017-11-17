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

def test_in_partition():
    """
    Test the in_partition method
    """
    dset = dataset_module.DataSet(logger)
    dset.partition(divisor=5, partitions=["one", "two", "three", "four", "five"])
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

def test_set_output_columns():
    """
    Test the set_output_columns method
    """
    dset = dataset_module.DataSet(logger)
    dset.ingest('data/test/test1.csv')
    # TBD

class TestPartition(unittest.TestCase):
    def test_partition(self):
        """
        Test the partition method and ensure it produces
        anticipated results when dataset outputs are requested
        """
        dset = dataset_module.DataSet(logger)
        #*** Check santity test, partitions list length different to divisor,
        #*** should raise a SystemExit:
        DIVISOR = 3
        PARTITIONS = ['A', 'B']
        with self.assertRaises(SystemExit):
            assert dset.partition(divisor=DIVISOR, partitions=PARTITIONS) == 0
        #*** TBD:
        
