#########
Configure
#########

The dataset class provides methods available through *project_policy.yaml*
to manipulate the ingested data so that it is suitable for processing.

*****************
Column Operations
*****************

Here are operations that can be performed on the dataset columns:

delete_columns
==============

TBD

duplicate_column
================

TBD

one_hot_encode
==============

Creates new column(s) with one hot encoded values. This is useful when you
have more than two result types in a column.

You need to specify a column that is used as the source for creating one-hot-encoded
columns. Note that this specified column is not updated.

Values in the column are listed that should be used to create new one-hot-encoded
columns. Note that the value is used as the column name.

Be careful to avoid column name collisions.

  Example:

  .. code-block:: text

        - one_hot_encode:
            - column: class
            - values:
                - Iris-setosa
                - Iris-versicolor
                - Iris-virginica

rescale
=======

set_output_columns
==================

Sets what columns are used as output data from dataset
(i.e. what columns contain the expected answer(s)
Pass it a list of output column names

  Example:

  .. code-block:: text

    - set_output_columns:
        - Iris-setosa
        - Iris-versicolor
        - Iris-virginica

translate
=========

TBD

trim_to_columns
===============

TBD

***********
Data Import
***********

ingest
======

TBD

***********
Data Export
***********

get_data
========

TBD

inputs_array
============

TBD

outputs_array
============

TBD

*******
General
*******

set_name
========

TBD

transform
=========

TBD


************
Partitioning
************

in_partition
============

TBD

partition
============

TBD

partition_sets
============

TBD

**************
Row Operations
**************

shuffle
=======

TBD

trim_to_rows
============

TBD

**********
Visibility
**********

display
=======

TBD


