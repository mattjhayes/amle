"""
AMLE dataset.py Unit Tests
"""

#*** Handle tests being in different directory branch to app code:
import sys
sys.path.insert(0, '../amle')

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
    assert dset.get_data() == [OrderedDict([('alice', '1'), ('bob', '1'), ('charlie', '1'), ('dave', 'foo')]),
                               OrderedDict([('alice', '3'), ('bob', '4'), ('charlie', '5'), ('dave', 'fighter')])]

def test_trim_to_columns():
    """
    Test the trim_to_columns method
    """
    dset = dataset_module.DataSet(logger)
    dset.ingest('data/test/test1.csv')
    dset.trim_to_columns(['alice', 'charlie'])
    logger.info("get_data=%s", dset.get_data())
    assert dset.get_data() == [OrderedDict([('alice', '1'), ('charlie', '1')]),
                               OrderedDict([('alice', '2'), ('charlie', '2')]),
                               OrderedDict([('alice', '3'), ('charlie', '5')])]



def test_set_output_columns():
    """
    Test the set_output_columns method
    """
    dset = dataset_module.DataSet(logger)
    dset.ingest('data/test/test1.csv')
    # TBD
