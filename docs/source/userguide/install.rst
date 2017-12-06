#######
Install
#######

WARNING: NOT TESTED YET

This guide is for installing on Ubuntu 16.04.2 LTS

********
Pre-Work
********

Ensure packages are up-to-date
==============================

.. code-block:: text

  sudo apt-get update
  sudo apt-get upgrade

***********************
Install Debian Packages
***********************

The following command installs these packages:

- pip (Python package manager)
- git (version control system)
- git flow (branching model for Git)
- python-pytest (used to run unit tests)
- python-yaml (YAML parser for Python)

.. code-block:: text

  sudo apt-get install python-pip git git-flow python-pytest python-yaml

***********************
Install Python Packages
***********************

The following command installs these Python packages:

- coloredlogs (Add colour to log entries in terminal output)
- voluptuous (data validation library)
- numpy (matrix and array library)
- matplotlib (optional)

.. code-block:: text

  pip install coloredlogs voluptuous numpy matplotlib
