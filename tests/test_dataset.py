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

def test_set_output_columns():
    """
    Test the set_output_columns method
    """
    dset = dataset_module.DataSet(logger)
    dset.ingest('data/test/test1.csv')
    # TBD
